#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nexar_depth_robust_v2.py

Entrenamiento Depth Multimodal (RGB + Depth) con paridad EXACTA al Fine-Tuning de V-JEPA,
pero con FUSIÓN DENSA RGB+DEPTH por patch y POOLING ESPACIAL APRENDIBLE.

Características:
  - Sampler clon: Hard Negatives + Jitter [min, max] idéntico a V-JEPA FT.
  - Objetivo: Loss sobre Max-Pooling temporal (Anytime Hard).
  - Arquitectura:
        * V-JEPA (Frozen/Unfrozen) → tokens por patch [B, steps, 256, D_rgb]
        * Depth tokenizer → tokens por patch [B, steps, 256, D_depth]
        * Fusión densa por patch + attention espacial (multi-head opcional) -> [B, T, D_fused]
        * Temporal Transformer Head (igual que FT) sobre [B, T, D_fused].

Uso:
  # Verificar shapes primero
  python train_nexar_depth_robust_v2.py --verify-shapes

  # Debug rápido
  python train_nexar_depth_robust_v2.py --train-csv ... --val-csv ... --debug

  # Entrenamiento completo
  python train_nexar_depth_robust_v2.py --train-csv ... --val-csv ... --spatial-heads 4
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
# 1. UTILIDADES
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
        clip = d_stack[nearest]
        clip = clip[..., None].astype(np.float32)
        return clip
    except Exception:
        return None


# ==============================================================================
# 2. DATASET (Lógica Clonada de V-JEPA FT)
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
# 2.1 COLLATE: RGB (processor) + Depth 256x256
# ==============================================================================
def make_collate_fn(processor: AutoVideoProcessor):
    """
    Collate que:
      - pasa RGB por AutoVideoProcessor (V-JEPA)
      - normaliza depth y la deja en [B,1,T,H,W] (resize lo hace DepthPatchTokenizer).
    """
    def collate_fn(batch):
        rgbs, depths, labels = zip(*batch)

        # RGB → processor (resize/normalize a lo que V-JEPA espera, p.ej. 256x256)
        inputs = processor(list(rgbs), return_tensors="pt")

        # Depth: lista de [T, H, W, 1] → tensor [B, T, H, W]
        d_np = np.stack(depths, axis=0)      # [B, T, H, W, 1]
        d_t = torch.from_numpy(d_np).float()
        d_t = d_t[..., 0]                    # [B, T, H, W]

        # Normalización física
        d_t = torch.clamp(d_t, min=0.0, max=150.0)
        d_t = torch.log1p(d_t) / 5.0         # log-scale

        # Añadir canal: [B, 1, T, H, W]
        d_t = d_t.unsqueeze(1).contiguous()

        labels_t = torch.tensor(labels, dtype=torch.float32)
        return inputs, d_t, labels_t

    return collate_fn


# ==============================================================================
# 3. MODELO: DepthPatchTokenizer + Fusión densa + SpatialPooler + Temporal Head
# ==============================================================================

class DepthPatchTokenizer(nn.Module):
    """
    Tokenizador de depth alineado con V-JEPA:
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

        x = self.patch_embed(depth_btchw)
        B, C2, S, Gh, Gw = x.shape
        assert S == self.steps and Gh == self.grid and Gw == self.grid

        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(B, self.steps, self.n_patches, C2)
        x = self.proj(x)
        return x


class DenseRGBDepthFusion(nn.Module):
    """
    Fusión por concat + proyección:
      rgb:   [B, steps, P, D_rgb]
      depth: [B, steps, P, D_depth]
      out:   [B, steps, P, D_fused]
    """
    def __init__(self, rgb_dim: int, depth_dim: int, fused_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(rgb_dim + depth_dim),
            nn.Linear(rgb_dim + depth_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, rgb_tokens, depth_tokens):
        x = torch.cat([rgb_tokens, depth_tokens], dim=-1)
        return self.proj(x)


class SpatialPooler(nn.Module):
    """
    Pooling espacial aprendible por frame con soporte multi-head.
    
    n_heads=1: Un solo MLP → score → softmax → weighted sum
    n_heads>1: Múltiples heads, cada uno con su propia atención espacial,
               luego se concatenan y proyectan.
    
    z_in:  [B, S, P, D]
    z_out: [B, S, D]
    """
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
            scores = self.mlp(z).squeeze(-1)
            weights = scores.softmax(dim=-1)
            z_frame = (weights.unsqueeze(-1) * z).sum(dim=2)
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


class VJEPA2DepthTransformerBinary(nn.Module):
    """
    Modelo con fusión densa RGB+Depth a nivel de patch.
    
    Flujo:
    1. V-JEPA → [B, steps, 256, D_rgb]
    2. DepthTokenizer → [B, steps, 256, D_depth]
    3. DenseFusion → [B, steps, 256, D_fused]
    4. SpatialPooler (aprendible) → [B, steps, D_fused]
    5. Interpolate → [B, T, D_fused]
    6. TemporalHead → [B, T] logits
    7. max_t → [B] (ANYTIME)
    """
    def __init__(
        self,
        hf_repo: str,
        depth_dim: int = 128,
        spatial_heads: int = 1,
        unfreeze_blocks: int = 0,
        encoder_ckpt: Optional[str] = None,
    ):
        super().__init__()

        # --- V-JEPA encoder ---
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

        if encoder_ckpt:
            sd = torch.load(encoder_ckpt, map_location="cpu")
            if "model_state" in sd:
                sd = sd["model_state"]
            self.vjepa2.load_state_dict(sd, strict=False)

        for p in self.vjepa2.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0:
            self._unfreeze_blocks(unfreeze_blocks)

        # --- Depth tokenizer ---
        self.depth_tokenizer = DepthPatchTokenizer(
            depth_dim=depth_dim,
            tubelet_size=self.tubelet,
            patch_size=self.patch_size,
            frames_per_clip=self.frames_per_clip,
            crop_size=self.crop_size,
            hidden_dim=128,
        )

        # --- Fusión densa + pooling espacial ---
        self.fused_dim = self.embed_dim + depth_dim

        self.fusion = DenseRGBDepthFusion(
            rgb_dim=self.embed_dim,
            depth_dim=depth_dim,
            fused_dim=self.fused_dim,
        )

        self.spatial_pooler = SpatialPooler(
            dim=self.fused_dim,
            hidden=256,
            n_heads=spatial_heads,
        )

        # --- Temporal head ---
        self.temporal_head = build_model(
            model_type="transformer",
            embed_dim=self.fused_dim,
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

        self._print_model_info(spatial_heads)

    def _print_model_info(self, spatial_heads: int):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] V-JEPA: D={self.embed_dim}, steps={self.steps}, "
              f"grid={self.grid}, patches={self.n_patches}")
        print(f"[MODEL] Fused dim: {self.fused_dim}, Spatial heads: {spatial_heads}")
        print(f"[MODEL] Params: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")

    def _unfreeze_blocks(self, n: int):
        blocks = getattr(self.vjepa2.encoder, "layers",
                         getattr(self.vjepa2.encoder, "layer", None))
        if blocks:
            total = len(blocks)
            for i, block in enumerate(blocks):
                if i >= total - n:
                    for p in block.parameters():
                        p.requires_grad = True
            print(f"[MODEL] Unfroze top {n} blocks of V-JEPA")

    def _rgb_tokens_spatio_temporal(self, pv):
        """Returns: [B, steps, n_patches, D_rgb]"""
        out = self.vjepa2(pixel_values_videos=pv, output_hidden_states=False)
        tokens = out.last_hidden_state
        B, L, D = tokens.shape

        expected_L = self.steps * self.n_patches
        if L != expected_L:
            tokens = tokens[:, 1:, :]
            L = tokens.shape[1]
            assert L == expected_L, f"Token length mismatch: {L} vs {expected_L}"

        x = tokens.view(B, self.steps, self.n_patches, D)
        return x

    def forward(self, pixel_values_videos, depth_videos, return_frame_logits=False):
        B = pixel_values_videos.shape[0]

        # 1) RGB tokens por patch
        z_rgb = self._rgb_tokens_spatio_temporal(pixel_values_videos)

        # 2) Depth tokens por patch
        z_depth = self.depth_tokenizer(depth_videos)

        assert z_depth.shape[1] == z_rgb.shape[1], f"steps mismatch"
        assert z_depth.shape[2] == z_rgb.shape[2], f"patch mismatch"

        # 3) Fusión densa por patch
        z_fused_patch = self.fusion(z_rgb, z_depth)

        # 4) Pool espacial APRENDIBLE
        z_steps = self.spatial_pooler(z_fused_patch)

        # 5) Interpolar steps -> T
        z_temporal = z_steps.transpose(1, 2)
        z_temporal = F.interpolate(
            z_temporal,
            size=self.frames_per_clip,
            mode="linear",
            align_corners=False,
        )
        z_temporal = z_temporal.transpose(1, 2)

        # 6) Temporal head
        T = z_temporal.shape[1]
        mask = torch.ones(B, T, device=z_temporal.device, dtype=torch.bool)
        logits_bt = self.temporal_head(z_temporal, mask=mask).squeeze(-1)

        if return_frame_logits:
            return logits_bt

        return logits_bt.max(dim=1).values


# ==============================================================================
# 4. VERIFY SHAPES
# ==============================================================================
def verify_shapes(hf_repo: str = "facebook/vjepa2-vitl-fpc16-256-ssv2", depth_dim: int = 128):
    """Verificación rápida de shapes antes de entrenar."""

    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE SHAPES - FUSIÓN DENSA + SPATIAL POOLER")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    B = 2
    # OJO: V-JEPA espera pixel_values_videos en formato [B, T, C, H, W]
    rgb   = torch.randn(B, 16, 3, 256, 256).to(device)  # [B,T,C,H,W]
    depth = torch.randn(B, 1, 16, 256, 256).to(device)  # [B,1,T,H,W]

    print(f"\n[INPUT]")
    print(f"  RGB:   {rgb.shape}   (esperado [B=2, T=16, C=3, 256, 256])")
    print(f"  Depth: {depth.shape} (esperado [B=2, 1, T=16, 256, 256])")

    model = VJEPA2DepthTransformerBinary(
        hf_repo=hf_repo,
        depth_dim=depth_dim,
        spatial_heads=1,
        unfreeze_blocks=0,
    ).to(device)

    # 1. RGB tokens
    z_rgb = model._rgb_tokens_spatio_temporal(rgb)
    print(f"\n[1] RGB tokens:   {z_rgb.shape}")
    print(f"    Esperado:     [B=2, steps=8, patches=256, D=1024]")
    assert z_rgb.shape == (B, 8, 256, 1024), f"❌ RGB incorrecto: {z_rgb.shape}"
    print("    ✅ OK")

    # 2. Depth tokens
    z_depth = model.depth_tokenizer(depth)
    print(f"\n[2] Depth tokens: {z_depth.shape}")
    print(f"    Esperado:     [B=2, steps=8, patches=256, D={depth_dim}]")
    assert z_depth.shape == (B, 8, 256, depth_dim), f"❌ Depth incorrecto: {z_depth.shape}"
    print("    ✅ OK")

    # 3. Fusión densa
    fused_dim = 1024 + depth_dim
    z_fused = model.fusion(z_rgb, z_depth)
    print(f"\n[3] Fused tokens: {z_fused.shape}")
    print(f"    Esperado:     [B=2, steps=8, patches=256, D={fused_dim}]")
    assert z_fused.shape == (B, 8, 256, fused_dim), f"❌ Fusión incorrecta: {z_fused.shape}"
    print("    ✅ OK")

    # 4. Spatial pooler
    z_steps = model.spatial_pooler(z_fused)
    print(f"\n[4] After pool:   {z_steps.shape}")
    print(f"    Esperado:     [B=2, steps=8, D={fused_dim}]")
    assert z_steps.shape == (B, 8, fused_dim), f"❌ Pool incorrecto: {z_steps.shape}"
    print("    ✅ OK")

    # 5. Forward completo
    logits = model(rgb, depth, return_frame_logits=True)
    print(f"\n[5] Logits:       {logits.shape}")
    print(f"    Esperado:     [B=2, T=16]")
    assert logits.shape == (B, 16), f"❌ Logits incorrectos: {logits.shape}"
    print("    ✅ OK")

    # 6. Max temporal
    final = model(rgb, depth, return_frame_logits=False)
    print(f"\n[6] Final (max):  {final.shape}")
    print(f"    Esperado:     [B=2]")
    assert final.shape == (B,), f"❌ Final incorrecto: {final.shape}"
    print("    ✅ OK")

    # 7. Memory check
    if device.type == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[MEM] Peak GPU: {mem_gb:.2f} GB (batch_size={B})")

    # 8. Test multi-head spatial pooler
    print("\n" + "-" * 40)
    print("Testing multi-head SpatialPooler (n_heads=4)...")
    pooler_mh = SpatialPooler(dim=fused_dim, hidden=256, n_heads=4).to(device)
    z_mh = pooler_mh(z_fused)
    assert z_mh.shape == (B, 8, fused_dim), f"❌ Multi-head pool incorrecto: {z_mh.shape}"
    print(f"    Multi-head output: {z_mh.shape} ✅ OK")

    print("\n" + "=" * 70)
    print("✅ TODAS LAS SHAPES CORRECTAS")
    print("=" * 70 + "\n")

    return True



# ==============================================================================
# 5. TRAINING LOOPS
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
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_samples = 0.0, 0
    agg_tp = agg_fp = agg_tn = agg_fn = 0

    all_probs = []
    all_labels = []

    for inputs, depth, labels in dataloader:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        depth = depth.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_bt = model(inputs["pixel_values_videos"], depth, return_frame_logits=True)
        logits_clip = logits_bt.max(dim=1).values
        loss = criterion(logits_clip, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        probs = torch.sigmoid(logits_clip)
        preds = (probs >= 0.5).long()
        y = labels.long()

        agg_tp += ((preds == 1) & (y == 1)).sum().item()
        agg_fp += ((preds == 1) & (y == 0)).sum().item()
        agg_tn += ((preds == 0) & (y == 0)).sum().item()
        agg_fn += ((preds == 0) & (y == 1)).sum().item()

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    mean_loss = total_loss / max(1, total_samples)
    acc, f1, prec, rec = calculate_global_metrics(agg_tp, agg_fp, agg_tn, agg_fn)

    # Calcular AP y AUC si sklearn está disponible
    ap, auc = 0.0, 0.0
    if SKLEARN_AVAILABLE and len(set(all_labels)) > 1:
        try:
            ap = average_precision_score(all_labels, all_probs)
            auc = roc_auc_score(all_labels, all_probs)
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
        "cm": (agg_tp, agg_fp, agg_tn, agg_fn),
    }


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    
    # Data
    ap.add_argument("--train-csv", type=str, default=None)
    ap.add_argument("--val-csv", type=str, default=None)
    ap.add_argument("--video-root", type=str, default=None)
    ap.add_argument("--depth-root", type=str, default=None)
    ap.add_argument("--out-dir", default="checkpoints_dense_fusion")

    # Model
    ap.add_argument("--hf-repo", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--depth-dim", type=int, default=128)
    ap.add_argument("--spatial-heads", type=int, default=1,
                    help="Número de heads en SpatialPooler (1=simple, 4=multi-head)")

    # Train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)  # Reducido por memoria
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
    ap.add_argument("--verify-shapes", action="store_true",
                    help="Solo verificar shapes y salir")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None)

    args = ap.parse_args()

    # --- Verify shapes mode ---
    if args.verify_shapes:
        verify_shapes(args.hf_repo, args.depth_dim)
        sys.exit(0)

    # --- Validar argumentos requeridos para entrenamiento ---
    if not all([args.train_csv, args.val_csv, args.video_root, args.depth_root]):
        print("[ERR] Para entrenar necesitas: --train-csv, --val-csv, --video-root, --depth-root")
        print("      O usa --verify-shapes para solo verificar shapes.")
        sys.exit(1)

    if args.train_jitter_max >= args.hist_s:
        print(f"[ERR] Jitter Max ({args.train_jitter_max}) >= Hist ({args.hist_s})")
        sys.exit(1)

    # --- W&B ---
    run_name = args.run_name or f"dense_h{args.spatial_heads}_d{args.depth_dim}"
    wandb.init(project="nexar_tfm_experiments", name=run_name, config=vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    fpc = getattr(AutoConfig.from_pretrained(args.hf_repo), "frames_per_clip", 16)

    jitter_range = (args.train_jitter_min, args.train_jitter_max)

    # --- Dataset ---
    if args.debug:
        print("[DEBUG] Usando subset en memoria...")
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
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # --- Model ---
    model = VJEPA2DepthTransformerBinary(
        hf_repo=args.hf_repo,
        depth_dim=args.depth_dim,
        spatial_heads=args.spatial_heads,
        unfreeze_blocks=args.unfreeze_blocks,
        encoder_ckpt=args.encoder_ckpt,
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

    # --- Resume ---
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

    # --- Training loop ---
    print(f"\n[TRAIN] Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"        Spatial heads: {args.spatial_heads}, Depth dim: {args.depth_dim}")
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
            save_path = Path(args.out_dir) / f"best_model_h{args.spatial_heads}.pt"
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
