#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_nexar_depth_mid_fusion_film.py

MID FUSION real (sin cambiar inputs de VJEPA2):
- Tokenizas Depth a nivel de patch/tubelet -> [B, steps, patches, Dd]
- Proyectas Depth -> (gamma,beta) por token RGB -> [B, L, Drgb]
- Inyectas (FiLM) ENTRE bloques del encoder usando hooks:
      x <- x * (1 + gamma) + beta
- Luego usas SOLO tokens RGB finales -> spatial pool -> temporal head -> anytime max

Esto SÍ es mid fusion porque depth afecta a los bloques superiores del backbone.
"""

import os
import sys
import csv
import argparse
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video

from transformers import AutoVideoProcessor, AutoConfig, VJEPA2Model

import wandb

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# OJO: adapta este import al path real de tu repo
from models import build_model

from thesis.utils.csv_utils import float_or_none, get_balanced_subset


# ==============================================================================
# 1) UTILS
# ==============================================================================

def _resolve_video_path(videos_root: Path, vid: str) -> Optional[Path]:
    valid_exts = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}
    matches = list(videos_root.rglob(f"{vid}.*"))
    for p in matches:
        if p.suffix in valid_exts:
            return p
    return None


def _resolve_depth_path(depth_root: Path, vid: str) -> Optional[Path]:
    p = depth_root / f"{vid}.npz"
    if p.exists():
        return p
    found = list(depth_root.rglob(f"{vid}.npz"))
    return found[0] if found else None


def _load_depth_clip_nearest(npz_path: Path, frame_indices: np.ndarray) -> Optional[np.ndarray]:
    try:
        data = np.load(str(npz_path))
        d_stack = data["depth"]
        d_indices = data["frame_idx"]
        if d_stack.shape[0] == 0:
            return None

        diffs = np.abs(d_indices[None, :] - frame_indices[:, None])
        nearest = diffs.argmin(axis=1)
        clip = d_stack[nearest]              # [T, H, W]
        clip = clip[..., None].astype(np.float32)  # [T, H, W, 1]
        return clip
    except Exception:
        return None


# ==============================================================================
# 2) DATASET (igual que tu robust)
# ==============================================================================

class NexarDepthRobustDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        data_rows: Optional[List] = None,
        videos_root: Path = None,
        depth_root: Path = None,
        frames_per_clip: int = 16,
        hist_s: float = 5.0,
        train_mode: bool = True,
        hard_neg_prob: float = 0.5,
        jitter_range: Tuple[float, float] = (0.0, 2.0),
        safe_margin_s: float = 2.0,
    ):
        super().__init__()
        self.videos_root = videos_root
        self.depth_root = depth_root
        self.frames_per_clip = int(frames_per_clip)
        self.hist_s = float(hist_s)

        self.train_mode = train_mode
        self.hard_neg_prob = hard_neg_prob
        self.jitter_range = jitter_range
        self.safe_margin_s = safe_margin_s

        rows_to_process = []
        if data_rows:
            rows_to_process = data_rows
        elif csv_path:
            with csv_path.open("r", encoding="utf-8") as f:
                rows_to_process = list(csv.DictReader(f))

        self.samples = []
        for row in rows_to_process:
            vid = row["id"].strip()
            if not vid:
                continue
            try:
                target = int(float(row["target"]))
            except:
                continue
            if target not in (0, 1):
                continue

            t_event = float_or_none(row.get("time_of_event", ""))
            vpath = _resolve_video_path(self.videos_root, vid)
            if vpath is None:
                continue
            dpath = _resolve_depth_path(self.depth_root, vid)

            self.samples.append({
                "id": vid,
                "vpath": vpath,
                "dpath": dpath,
                "target": target,
                "time_of_event": t_event,
            })

        print(f"[DATASET] {len(self.samples)} vídeos. Mode={'TRAIN (Hard)' if train_mode else 'VAL'}")

    def __len__(self) -> int:
        return len(self.samples)

    def _sec_to_idx(self, center_sec, duration, fps, T):
        start_sec = max(0.0, center_sec - self.hist_s)
        end_sec = min(duration, center_sec)

        if end_sec <= start_sec:
            start_sec = 0.0
            end_sec = duration

        start_idx = int(round(start_sec * fps))
        end_idx = int(round(end_sec * fps))
        start_idx = max(0, min(start_idx, T - 1))
        end_idx = max(start_idx, min(end_idx, T - 1))

        if start_idx == end_idx:
            idx = torch.full((self.frames_per_clip,), start_idx, dtype=torch.long)
        else:
            idx = torch.linspace(start_idx, end_idx, steps=self.frames_per_clip).long()
            idx = torch.clamp(idx, 0, T - 1)
        return idx

    def _get_indices_robust(self, T, fps, t_event, target_video):
        duration = T / max(fps, 1e-6)
        final_label = target_video

        if not self.train_mode:
            if target_video == 1 and t_event is not None:
                center_sec = t_event
            else:
                center_sec = 0.5 * duration
            return self._sec_to_idx(center_sec, duration, fps, T), final_label

        use_positive_sample = False

        if target_video == 1 and t_event is not None:
            if torch.rand(1).item() > self.hard_neg_prob:
                use_positive_sample = True
                final_label = 1
            else:
                use_positive_sample = False
                final_label = 0
        else:
            use_positive_sample = False
            final_label = 0

        if use_positive_sample:
            low, high = self.jitter_range
            offset = torch.empty(1).uniform_(low, high).item()
            center_sec = t_event + offset
        else:
            valid_intervals = []
            if t_event is not None:
                end_z1 = t_event - self.safe_margin_s
                if end_z1 > self.hist_s:
                    valid_intervals.append((self.hist_s, end_z1))
                start_z2 = t_event + self.safe_margin_s + self.hist_s
                if start_z2 < duration:
                    valid_intervals.append((start_z2, duration))
            else:
                if duration > self.hist_s:
                    valid_intervals.append((self.hist_s, duration))

            if valid_intervals:
                idx = torch.randint(len(valid_intervals), (1,)).item()
                s, e = valid_intervals[idx]
                center_sec = torch.empty(1).uniform_(s, e).item()
            else:
                center_sec = t_event if target_video == 1 else 0.5 * duration
                if target_video == 1:
                    final_label = 1

        return self._sec_to_idx(center_sec, duration, fps, T), final_label

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        try:
            video, _, info = read_video(str(sample["vpath"]), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except:
            video = torch.zeros((self.frames_per_clip, 224, 224, 3), dtype=torch.uint8)
            fps = 30.0

        if video.numel() == 0:
            video = torch.zeros((self.frames_per_clip, 224, 224, 3), dtype=torch.uint8)

        idxs, label_clip = self._get_indices_robust(
            video.shape[0], fps, sample["time_of_event"], sample["target"]
        )

        rgb_clip = video[idxs].numpy()

        dpath = sample["dpath"]
        depth_clip = None
        if dpath:
            depth_clip = _load_depth_clip_nearest(dpath, idxs.numpy())
        if depth_clip is None:
            depth_clip = np.zeros((self.frames_per_clip, 224, 224, 1), dtype=np.float32)

        return rgb_clip, depth_clip, label_clip


# ==============================================================================
# 2.1) COLLATE
# ==============================================================================
def make_collate_fn(processor: AutoVideoProcessor):
    def collate_fn(batch):
        rgbs, depths, labels = zip(*batch)

        inputs = processor(list(rgbs), return_tensors="pt")

        d_np = np.stack(depths, axis=0)      # [B, T, H, W, 1]
        d_t = torch.from_numpy(d_np).float()
        d_t = d_t[..., 0]                    # [B, T, H, W]

        # normalización depth (mantén esto estable, si no el adapter rompe)
        d_t = torch.clamp(d_t, min=0.0, max=150.0)
        d_t = torch.log1p(d_t) / 5.0

        d_t = d_t.unsqueeze(1).contiguous()  # [B, 1, T, H, W]

        labels_t = torch.tensor(labels, dtype=torch.float32)
        return inputs, d_t, labels_t

    return collate_fn


# ==============================================================================
# 3) DEPTH TOKENIZER (igual que el tuyo)
# ==============================================================================

class DepthPatchTokenizer(nn.Module):
    """
    Input:  [B, 1, T, H, W]
    Output: [B, steps, grid^2, D_depth]
    """
    def __init__(
        self,
        depth_dim: int = 128,
        tubelet_size: int = 2,
        patch_size: int = 16,
        frames_per_clip: int = 16,
        crop_size: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.frames_per_clip = frames_per_clip
        self.crop_size = crop_size

        self.steps = frames_per_clip // tubelet_size
        self.grid = crop_size // patch_size
        self.n_patches = self.grid ** 2

        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=0,
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, depth_dim),
            nn.GELU(),
            nn.Linear(depth_dim, depth_dim),
        )

    def forward(self, depth_btchw: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = depth_btchw.shape

        if (T != self.frames_per_clip) or (H != self.crop_size) or (W != self.crop_size):
            depth_btchw = F.interpolate(
                depth_btchw,
                size=(self.frames_per_clip, self.crop_size, self.crop_size),
                mode="trilinear",
                align_corners=False,
            )

        x = self.patch_embed(depth_btchw)          # [B, hidden, steps, grid, grid]
        B, C2, S, Gh, Gw = x.shape
        assert S == self.steps and Gh == self.grid and Gw == self.grid

        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, steps, grid, grid, hidden]
        x = x.reshape(B, self.steps, self.n_patches, C2)
        x = self.proj(x)
        return x


# ==============================================================================
# 4) MID-FUSION ADAPTER: DepthFiLM
# ==============================================================================

class DepthFiLM(nn.Module):
    """
    depth_tokens_flat: [B, L, Dd]  -> gamma,beta: [B, L, Drgb]
    Importante: init ~ 0 para empezar identidad (si no, te rompe el backbone frozen).
    """
    def __init__(self, depth_dim: int, rgb_dim: int, hidden: int = 256, scale: float = 0.10):
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.LayerNorm(depth_dim),
            nn.Linear(depth_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * rgb_dim),
        )
        # identidad al inicio
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_depth_flat: torch.Tensor):
        gb = self.net(z_depth_flat)  # [B, L, 2*Drgb]
        gamma, beta = gb.chunk(2, dim=-1)
        return self.scale * gamma, self.scale * beta


# ==============================================================================
# 5) SPATIAL POOLER (igual que el tuyo)
# ==============================================================================

class SpatialPooler(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, n_heads: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim

        if n_heads == 1:
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.norm = nn.LayerNorm(dim)
            head_hidden = max(hidden // n_heads, 64)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(n_heads)
            ])
            self.combine = nn.Linear(dim * n_heads, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, S, P, D = z.shape

        if self.n_heads == 1:
            scores = self.mlp(z).squeeze(-1)       # [B, S, P]
            weights = scores.softmax(dim=-1)
            z_frame = (weights.unsqueeze(-1) * z).sum(dim=2)  # [B, S, D]
        else:
            z_norm = self.norm(z)
            head_outputs = []
            for head in self.heads:
                scores = head(z_norm).squeeze(-1)
                weights = scores.softmax(dim=-1)
                h_out = (weights.unsqueeze(-1) * z).sum(dim=2)
                head_outputs.append(h_out)
            z_frame = self.combine(torch.cat(head_outputs, dim=-1))

        return z_frame


# ==============================================================================
# 6) MODELO MID FUSION: hooks en encoder.layer[i]
# ==============================================================================

def _parse_inject_layers(s: str, n_layers: int) -> List[int]:
    """
    Permite: "-4,-3,-2,-1" o "20,21,22,23"
    """
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    idxs = []
    for p in parts:
        i = int(p)
        if i < 0:
            i = n_layers + i
        idxs.append(i)
    idxs = sorted(set([i for i in idxs if 0 <= i < n_layers]))
    return idxs


class VJEPA2DepthMidFiLMBinary(nn.Module):
    """
    MID fusion:
      - Calcula depth tokens
      - Genera (gamma,beta) por token y lo inyecta en encoder.layer[i] via hooks
      - Usa tokens finales RGB para la cabeza (sin concat late)
    """
    def __init__(
        self,
        hf_repo: str,
        depth_dim: int = 128,
        spatial_heads: int = 1,
        unfreeze_blocks: int = 0,
        encoder_ckpt: Optional[str] = None,
        inject_layers: str = "-4,-3,-2,-1",
        film_hidden: int = 256,
        film_scale: float = 0.10,
        per_layer_adapters: bool = True,
    ):
        super().__init__()

        self.vjepa2 = VJEPA2Model.from_pretrained(hf_repo)
        cfg = self.vjepa2.config

        self.embed_dim = cfg.hidden_size
        self.frames_per_clip = getattr(cfg, "frames_per_clip", 16)
        self.crop_size = getattr(cfg, "crop_size", 256)
        self.patch_size = getattr(cfg, "patch_size", 16)
        self.grid = self.crop_size // self.patch_size
        self.tubelet = cfg.tubelet_size
        self.steps = self.frames_per_clip // self.tubelet
        self.n_patches = self.grid ** 2
        self.expected_L = self.steps * self.n_patches

        if encoder_ckpt:
            sd = torch.load(encoder_ckpt, map_location="cpu")
            if "model_state" in sd:
                sd = sd["model_state"]
            self.vjepa2.load_state_dict(sd, strict=False)

        # Freeze backbone por defecto
        for p in self.vjepa2.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0:
            self._unfreeze_blocks(unfreeze_blocks)

        # Depth tokenizer
        self.depth_tokenizer = DepthPatchTokenizer(
            depth_dim=depth_dim,
            tubelet_size=self.tubelet,
            patch_size=self.patch_size,
            frames_per_clip=self.frames_per_clip,
            crop_size=self.crop_size,
            hidden_dim=128,
        )

        # Adapter(s)
        n_layers = len(self.vjepa2.encoder.layer)
        self.inject_idxs = _parse_inject_layers(inject_layers, n_layers)
        if len(self.inject_idxs) == 0:
            raise ValueError("inject_layers vacío -> eso NO es mid fusion. Pon algo tipo -4,-3,-2,-1.")

        self.per_layer_adapters = bool(per_layer_adapters)
        if self.per_layer_adapters:
            self.film = nn.ModuleDict({
                str(i): DepthFiLM(depth_dim, self.embed_dim, hidden=film_hidden, scale=film_scale)
                for i in self.inject_idxs
            })
        else:
            self.film_shared = DepthFiLM(depth_dim, self.embed_dim, hidden=film_hidden, scale=film_scale)

        # Spatial pool + temporal head (como tu FT)
        self.spatial_pooler = SpatialPooler(dim=self.embed_dim, hidden=256, n_heads=spatial_heads)

        self.temporal_head = build_model(
            model_type="transformer",
            embed_dim=self.embed_dim,
            n_windows=1,
            d_model=384,
            n_heads=6,
            num_layers=4,
            ff_dim=1536,
            dropout=0.10,
            pe_dropout=0.05,
            attn_dropout=0.05,
            causal=True,
        )

        # Hooks
        self._film_ctx = None
        self._hook_handles = []
        self._register_hooks()

        self._print_model_info(spatial_heads, film_hidden, film_scale)

    def _print_model_info(self, spatial_heads, film_hidden, film_scale):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] V-JEPA: D={self.embed_dim}, steps={self.steps}, grid={self.grid}, patches={self.n_patches}")
        print(f"[MODEL] MID-FUSION FiLM layers: {self.inject_idxs} | per-layer={self.per_layer_adapters}")
        print(f"[MODEL] FiLM hidden={film_hidden}, scale={film_scale} | Spatial heads={spatial_heads}")
        print(f"[MODEL] Params: {trainable:,} / {total:,} trainable ({100*trainable/max(1,total):.1f}%)")

    def _unfreeze_blocks(self, n: int):
        blocks = self.vjepa2.encoder.layer
        total = len(blocks)
        for i, block in enumerate(blocks):
            if i >= total - n:
                for p in block.parameters():
                    p.requires_grad = True
        print(f"[MODEL] Unfroze top {n} blocks of V-JEPA")

    def _register_hooks(self):
        # limpia si re-creas
        for h in self._hook_handles:
            try:
                h.remove()
            except:
                pass
        self._hook_handles = []

        for i in self.inject_idxs:
            layer = self.vjepa2.encoder.layer[i]
            handle = layer.register_forward_hook(self._make_film_hook(i))
            self._hook_handles.append(handle)

    def _make_film_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            ctx = self._film_ctx
            if ctx is None:
                return output

            pack = ctx.get(layer_idx, None)
            if pack is None:
                return output

            gamma_no, beta_no, gamma_cls, beta_cls = pack

            # output puede ser tensor o tuple (hs, attn,...)
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = None

            B, L, D = hs.shape

            if gamma_cls is not None and L == gamma_cls.shape[1]:
                gamma, beta = gamma_cls, beta_cls
            elif L == gamma_no.shape[1]:
                gamma, beta = gamma_no, beta_no
            else:
                # mismatch raro: mejor no tocar que romper shapes
                return output

            gamma = gamma.to(device=hs.device, dtype=hs.dtype)
            beta  = beta.to(device=hs.device, dtype=hs.dtype)

            hs2 = hs * (1.0 + gamma) + beta

            if rest is None:
                return hs2
            return (hs2,) + rest

        return hook

    def _compute_film_context(self, depth_videos: torch.Tensor):
        """
        Calcula gamma/beta por layer, con y sin CLS.
        """
        z_depth = self.depth_tokenizer(depth_videos)                     # [B, steps, P, Dd]
        B, S, P, Dd = z_depth.shape
        assert S * P == self.expected_L, "Depth tokens L mismatch"
        z_flat = z_depth.reshape(B, self.expected_L, Dd)                 # [B, L, Dd]

        ctx = {}
        for i in self.inject_idxs:
            if self.per_layer_adapters:
                gamma, beta = self.film[str(i)](z_flat)
            else:
                gamma, beta = self.film_shared(z_flat)

            # con CLS (si existe)
            gamma_cls = beta_cls = None
            gamma0 = torch.zeros((B, 1, self.embed_dim), device=gamma.device, dtype=gamma.dtype)
            beta0  = torch.zeros((B, 1, self.embed_dim), device=beta.device, dtype=beta.dtype)
            gamma_cls = torch.cat([gamma0, gamma], dim=1)                # [B, L+1, D]
            beta_cls  = torch.cat([beta0,  beta ], dim=1)

            ctx[i] = (gamma, beta, gamma_cls, beta_cls)

        return ctx

    def _rgb_tokens_final(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta VJEPA2 forward normal (los hooks hacen mid fusion).
        Devuelve tokens finales SIN CLS, reshapeables a [B, steps, P, D].
        """
        out = self.vjepa2(pixel_values_videos=pixel_values_videos, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B, L or L+1, D]

        B, L, D = tokens.shape
        if L == self.expected_L + 1:
            tokens = tokens[:, 1:, :]
            L = tokens.shape[1]
        if L != self.expected_L:
            raise RuntimeError(f"Token length mismatch: got L={L}, expected {self.expected_L} (or +1 CLS).")

        z = tokens.view(B, self.steps, self.n_patches, D)
        return z

    def forward(self, pixel_values_videos, depth_videos, return_frame_logits: bool = False):
        # 1) prepara ctx para hooks (gamma/beta por layer)
        self._film_ctx = self._compute_film_context(depth_videos)

        # 2) corre backbone (hooks aplican FiLM entre capas)
        z_rgb = self._rgb_tokens_final(pixel_values_videos)  # [B, steps, P, D]

        # 3) limpiar ctx para no contaminar otros forwards
        self._film_ctx = None

        # 4) spatial pooling -> [B, steps, D]
        z_steps = self.spatial_pooler(z_rgb)

        # 5) interpolate steps -> T
        z_temporal = z_steps.transpose(1, 2)  # [B, D, steps]
        z_temporal = F.interpolate(z_temporal, size=self.frames_per_clip, mode="linear", align_corners=False)
        z_temporal = z_temporal.transpose(1, 2)  # [B, T, D]

        # 6) temporal head -> logits [B, T]
        B = z_temporal.shape[0]
        T = z_temporal.shape[1]
        mask = torch.ones(B, T, device=z_temporal.device, dtype=torch.bool)
        logits_bt = self.temporal_head(z_temporal, mask=mask).squeeze(-1)

        if return_frame_logits:
            return logits_bt
        return logits_bt.max(dim=1).values


# ==============================================================================
# 7) VERIFY SHAPES
# ==============================================================================
@torch.no_grad()
def verify_shapes(hf_repo: str, depth_dim: int, inject_layers: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    rgb   = torch.randn(B, 16, 3, 256, 256, device=device)
    depth = torch.randn(B, 1, 16, 256, 256, device=device)

    model = VJEPA2DepthMidFiLMBinary(
        hf_repo=hf_repo,
        depth_dim=depth_dim,
        spatial_heads=1,
        unfreeze_blocks=0,
        inject_layers=inject_layers,
        film_hidden=256,
        film_scale=0.10,
        per_layer_adapters=True,
    ).to(device)

    logits = model(rgb, depth, return_frame_logits=True)
    assert logits.shape == (B, 16), logits.shape
    final = model(rgb, depth, return_frame_logits=False)
    assert final.shape == (B,), final.shape

    print("[VERIFY] OK shapes. logits_bt:", logits.shape, "final:", final.shape)
    if device.type == "cuda":
        print("[VERIFY] peak GB:", torch.cuda.max_memory_allocated() / 1e9)


# ==============================================================================
# 8) TRAINING LOOPS (igual que tuyo)
# ==============================================================================
def calculate_global_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(1, total)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, f1, prec, rec


def train_one_epoch(model, dataloader, device, optimizer, criterion, scaler, epoch, clip_norm):
    model.train()
    total_loss, total_samples = 0.0, 0
    agg_tp = agg_fp = agg_tn = agg_fn = 0

    pbar = tqdm(dataloader, desc=f"Train E{epoch:02d}", unit="bt") if tqdm else dataloader

    for inputs, depth, labels in pbar:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        depth = depth.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits_bt = model(inputs["pixel_values_videos"], depth, return_frame_logits=True)
            logits_clip = logits_bt.max(dim=1).values
            loss = criterion(logits_clip, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        B = labels.size(0)
        total_loss += loss.item() * B
        total_samples += B

        with torch.no_grad():
            preds = (torch.sigmoid(logits_clip) >= 0.5).long()
            y = labels.long()
            agg_tp += ((preds == 1) & (y == 1)).sum().item()
            agg_fp += ((preds == 1) & (y == 0)).sum().item()
            agg_tn += ((preds == 0) & (y == 0)).sum().item()
            agg_fn += ((preds == 0) & (y == 1)).sum().item()

        if tqdm and pbar is not dataloader:
            acc = (agg_tp + agg_tn) / max(1, total_samples)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.1%}")

    mean_loss = total_loss / max(1, total_samples)
    acc, f1, prec, rec = calculate_global_metrics(agg_tp, agg_fp, agg_tn, agg_fn)
    return mean_loss, acc * 100.0, f1


@torch.no_grad()
@torch.no_grad()
def evaluate(model, dataloader, device, criterion, thr_grid=None):
    """
    Eval robusta:
      - loss (igual que antes)
      - métricas @thr=0.5
      - best_f1 + best_thr haciendo sweep de umbrales (recomendado si usas pos_weight)

    Returns keys:
      loss, acc, f1, prec, rec, ap, auc, cm,
      best_f1, best_thr, best_prec, best_rec, cm_best
    """
    model.eval()
    total_loss, total_samples = 0.0, 0

    probs_all = []
    labels_all = []

    for inputs, depth, labels in dataloader:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        depth = depth.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_bt = model(inputs["pixel_values_videos"], depth, return_frame_logits=True)
        logits_clip = logits_bt.max(dim=1).values  # [B]
        loss = criterion(logits_clip, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        probs = torch.sigmoid(logits_clip)  # [B]
        probs_all.append(probs.detach().cpu())
        labels_all.append(labels.detach().cpu())

    mean_loss = total_loss / max(1, total_samples)

    probs_all = torch.cat(probs_all, dim=0).float()     # [N]
    labels_all = torch.cat(labels_all, dim=0).float()   # [N]
    y = (labels_all > 0.5).long()                       # [N] {0,1}

    # -------------------------
    # métricas @ thr=0.5 (como antes)
    # -------------------------
    thr_default = 0.5
    preds = (probs_all >= thr_default).long()

    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # -------------------------
    # best-F1 por sweep de umbral (esto es lo importante)
    # -------------------------
    if thr_grid is None:
        # grid decente sin ser caro (puedes subir a 101 puntos si quieres)
        thr_grid = torch.linspace(0.05, 0.95, steps=19)

    thr_grid = thr_grid.float()

    # vectorizado: [K,N]
    P = probs_all.unsqueeze(0) >= thr_grid.unsqueeze(1)
    Y = y.unsqueeze(0).expand_as(P)

    tp_k = (P & (Y == 1)).sum(dim=1).float()
    fp_k = (P & (Y == 0)).sum(dim=1).float()
    fn_k = ((~P) & (Y == 1)).sum(dim=1).float()
    tn_k = ((~P) & (Y == 0)).sum(dim=1).float()

    prec_k = tp_k / (tp_k + fp_k + 1e-12)
    rec_k  = tp_k / (tp_k + fn_k + 1e-12)
    f1_k   = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-12)

    best_idx = torch.argmax(f1_k).item()
    best_thr = float(thr_grid[best_idx].item())
    best_f1  = float(f1_k[best_idx].item())
    best_prec = float(prec_k[best_idx].item())
    best_rec  = float(rec_k[best_idx].item())

    tp_b = int(tp_k[best_idx].item())
    fp_b = int(fp_k[best_idx].item())
    tn_b = int(tn_k[best_idx].item())
    fn_b = int(fn_k[best_idx].item())

    # -------------------------
    # AP / AUC si sklearn disponible
    # -------------------------
    ap, auc = 0.0, 0.0
    if SKLEARN_AVAILABLE and len(torch.unique(y)) > 1:
        try:
            ap = average_precision_score(y.numpy().tolist(), probs_all.numpy().tolist())
            auc = roc_auc_score(y.numpy().tolist(), probs_all.numpy().tolist())
        except:
            pass

    return {
        "loss": mean_loss,
        "acc": acc * 100.0,
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "ap": ap,
        "auc": auc,
        "cm": (tp, fp, tn, fn),              # @0.5

        "best_f1": best_f1,
        "best_thr": best_thr,
        "best_prec": best_prec,
        "best_rec": best_rec,
        "cm_best": (tp_b, fp_b, tn_b, fn_b), # @best_thr
    }



# ==============================================================================
# 9) MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train-csv", type=str, default=None)
    ap.add_argument("--val-csv", type=str, default=None)
    ap.add_argument("--video-root", type=str, default=None)
    ap.add_argument("--depth-root", type=str, default=None)
    ap.add_argument("--out-dir", default="checkpoints_mid_film")

    # Model
    ap.add_argument("--hf-repo", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--depth-dim", type=int, default=128)
    ap.add_argument("--spatial-heads", type=int, default=1)

    # Mid fusion params
    ap.add_argument("--inject-layers", type=str, default="-4,-3,-2,-1",
                    help="Ej: '-4,-3,-2,-1' o '20,21,22,23'")
    ap.add_argument("--film-hidden", type=int, default=256)
    ap.add_argument("--film-scale", type=float, default=0.10)
    ap.add_argument("--film-per-layer", action="store_true",
                    help="Si se pone, usa un adapter distinto por capa (recomendado).")

    # Train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")

    # Robustness
    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--hard-neg-prob", type=float, default=0.5)
    ap.add_argument("--train-jitter-min", type=float, default=0.0)
    ap.add_argument("--train-jitter-max", type=float, default=2.0)

    # Utils
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--verify-shapes", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None)

    args = ap.parse_args()

    if args.verify_shapes:
        verify_shapes(args.hf_repo, args.depth_dim, args.inject_layers)
        sys.exit(0)

    if not all([args.train_csv, args.val_csv, args.video_root, args.depth_root]):
        print("[ERR] Para entrenar necesitas: --train-csv, --val-csv, --video-root, --depth-root")
        print("      O usa --verify-shapes para solo verificar.")
        sys.exit(1)

    if args.train_jitter_max >= args.hist_s:
        print(f"[ERR] Jitter Max ({args.train_jitter_max}) >= Hist ({args.hist_s})")
        sys.exit(1)

    run_name = args.run_name or f"midfilm_L{args.inject_layers}_s{args.film_scale}_d{args.depth_dim}"
    wandb.init(project="nexar_tfm_experiments", name=run_name, config=vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    fpc = getattr(AutoConfig.from_pretrained(args.hf_repo), "frames_per_clip", 16)
    jitter_range = (args.train_jitter_min, args.train_jitter_max)

    if args.debug:
        train_rows = get_balanced_subset(Path(args.train_csv), 80)
        val_rows = get_balanced_subset(Path(args.val_csv), 30)
        train_ds = NexarDepthRobustDataset(
            data_rows=train_rows,
            videos_root=Path(args.video_root),
            depth_root=Path(args.depth_root),
            frames_per_clip=fpc,
            hist_s=args.hist_s,
            train_mode=True,
            hard_neg_prob=args.hard_neg_prob,
            jitter_range=jitter_range,
        )
        val_ds = NexarDepthRobustDataset(
            data_rows=val_rows,
            videos_root=Path(args.video_root),
            depth_root=Path(args.depth_root),
            frames_per_clip=fpc,
            hist_s=args.hist_s,
            train_mode=False,
        )
    else:
        train_ds = NexarDepthRobustDataset(
            csv_path=Path(args.train_csv),
            videos_root=Path(args.video_root),
            depth_root=Path(args.depth_root),
            frames_per_clip=fpc,
            hist_s=args.hist_s,
            train_mode=True,
            hard_neg_prob=args.hard_neg_prob,
            jitter_range=jitter_range,
        )
        val_ds = NexarDepthRobustDataset(
            csv_path=Path(args.val_csv),
            videos_root=Path(args.video_root),
            depth_root=Path(args.depth_root),
            frames_per_clip=fpc,
            hist_s=args.hist_s,
            train_mode=False,
        )

    collate = make_collate_fn(processor)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=True)

    model = VJEPA2DepthMidFiLMBinary(
        hf_repo=args.hf_repo,
        depth_dim=args.depth_dim,
        spatial_heads=args.spatial_heads,
        unfreeze_blocks=args.unfreeze_blocks,
        encoder_ckpt=args.encoder_ckpt,
        inject_layers=args.inject_layers,
        film_hidden=args.film_hidden,
        film_scale=args.film_scale,
        per_layer_adapters=args.film_per_layer,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    best_val_f1 = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"[RESUME] Cargando {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("val_f1", 0.0)
    elif args.init_from and os.path.isfile(args.init_from):
        print(f"[INIT] Cargando pesos de {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device)
        st = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(st, strict=False)

    print(f"\n[TRAIN] Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"        inject_layers: {args.inject_layers} | film_scale: {args.film_scale} | unfreeze: {args.unfreeze_blocks}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t_loss, t_acc, t_f1 = train_one_epoch(
            model, train_ld, device, optimizer, criterion, scaler, epoch, args.grad_clip
        )
        val_metrics = evaluate(model, val_ld, device, criterion)

        v_loss = val_metrics["loss"]
        v_f1 = val_metrics["f1"]
        v_prec = val_metrics["prec"]
        v_rec = val_metrics["rec"]
        v_ap = val_metrics["ap"]
        v_auc = val_metrics["auc"]
        tp, fp, tn, fn = val_metrics["cm"]

        print(
            f"[E{epoch:02d}] T_Loss:{t_loss:.3f} T_F1:{t_f1:.3f} | "
            f"V_Loss:{v_loss:.3f} V_F1:{v_f1:.3f} (P:{v_prec:.2f}/R:{v_rec:.2f})"
        )
        print(f"       AP:{v_ap:.3f} AUC:{v_auc:.3f} | CM: [TN:{tn} FP:{fp}] [FN:{fn} TP:{tp}]")

        wandb.log({
            "epoch": epoch,
            "train/loss": t_loss,
            "train/acc": t_acc,
            "train/f1": t_f1,
            "val/loss": v_loss,
            "val/acc": val_metrics["acc"],
            "val/f1": v_f1,
            "val/prec": v_prec,
            "val/rec": v_rec,
            "val/ap": v_ap,
            "val/auc": v_auc,
            "val/cm/tp": tp,
            "val/cm/fp": fp,
            "val/cm/tn": tn,
            "val/cm/fn": fn,
            "best_val_f1": max(best_val_f1, v_f1),
        })

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            save_path = Path(args.out_dir) / f"best_midfilm_{args.inject_layers.replace(',','_')}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_f1": v_f1,
                "val_ap": v_ap,
                "args": vars(args),
            }, save_path)
            print(f"       ✅ Saved Best F1: {v_f1:.3f} → {save_path}")

    print("\n" + "=" * 60)
    print(f"✅ Training complete. Best F1: {best_val_f1:.3f}")
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    main()
