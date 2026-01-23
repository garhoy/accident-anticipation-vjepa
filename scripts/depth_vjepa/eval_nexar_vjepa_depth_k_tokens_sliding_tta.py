#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_nexar_vjepa_depth_k_tokens_sliding_tta.py

Evaluación REAL-WORLD (sliding window) para:
    V-JEPA 2 + DepthAnything3 (npz) + Patchwise Fusion + (K tokens por step)

Soporta checkpoints entrenados con:
  - K=1  (pooling clásico 1 token por step)
  - K>1  (MultiQuerySpatialPooler: K "glimpses" por step)

Métricas:
  - AP_anytime, AUC_anytime
  - Acc, Prec, Rec, F1 @0.5
  - mTTA@thr (ventanas deslizantes)
  - Recall_before_event@thr
  - Bloque @R80: thr_R80, Acc/F1/mTTA/Recall_before_event

Uso típico:
  CUDA_VISIBLE_DEVICES=0 python eval_nexar_vjepa_depth_k_tokens_sliding_tta.py \
    --checkpoint checkpoints_depth_k_tokens/best_model_K4_max.pt \
    --csv "/home/ander/V-JEPA 2/data/metadata/Nexar/val.csv" \
    --video-root "/home/ander/V-JEPA 2/data/raw/Nexar" \
    --depth-root "/home/ander/BADAS-Open/data/processed/Nexar_DA3_Tensors" \
    --hist-s 5.0 --stride-s 0.5 --agg max --batch-size 16 \
    --tta-thr 0.5
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_video
from transformers import AutoConfig, AutoVideoProcessor, VJEPA2Model

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Importa el factory del temporal head (igual que en train)
from models import build_model

from thesis.utils.csv_utils import float_or_none


# ==========================================================
# 0) HELPERS
# ==========================================================

def get_model_size_mb(model: nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def read_csv_with_event(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = r.get("id", "").strip()
            if not vid:
                continue
            try:
                t = int(float(r.get("target", "0")))
            except Exception:
                continue
            tev = float_or_none(r.get("time_of_event", ""))
            rows.append({"id": vid, "target": t, "time_of_event": tev})
    return rows


# ==========================================================
# 1) INDEXADO DE VÍDEOS Y DEPTH
# ==========================================================

def index_videos(video_root: Path) -> Dict[str, Path]:
    valid_exts = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}
    video_map: Dict[str, Path] = {}
    print(f"[INDEX] Indexando vídeos en {video_root} ...")
    for p in video_root.rglob("*"):
        if p.suffix in valid_exts:
            video_map[p.stem] = p
    print(f"[INDEX] Vídeos indexados: {len(video_map)}")
    return video_map

def index_depth(depth_root: Path) -> Dict[str, Path]:
    depth_map: Dict[str, Path] = {}
    print(f"[INDEX] Indexando depth npz en {depth_root} ...")
    for p in depth_root.rglob("*.npz"):
        depth_map[p.stem] = p
    print(f"[INDEX] Depth indexados: {len(depth_map)}")
    return depth_map

def load_depth_clip_nearest(npz_path: Path,
                            frame_indices: np.ndarray,
                            frames_per_clip: int) -> np.ndarray:
    """
    Devuelve [T,H,W,1] float32. Si falla, zeros.
    """
    try:
        data = np.load(str(npz_path))
        d_stack = data["depth"]       # [N,H,W]
        d_idx = data["frame_idx"]     # [N]
        if d_stack.shape[0] == 0:
            raise RuntimeError("Empty depth stack")

        diffs = np.abs(d_idx[None, :] - frame_indices[:, None])
        nearest = diffs.argmin(axis=1)        # [T]
        clip = d_stack[nearest]               # [T,H,W]
        clip = clip[..., None].astype(np.float32)
        return clip
    except Exception:
        return np.zeros((frames_per_clip, 224, 224, 1), dtype=np.float32)


# ==========================================================
# 2) TTA + THR@R80
# ==========================================================

def compute_tta_for_video(
    window_times: List[float],
    clip_probs: List[float],
    time_of_event: float,
    thr: float,
) -> Tuple[float, bool]:
    """
    t_detect = min t_window s.t. prob>=thr y t_window<=t_event
    TTA = max(t_event - t_detect, 0)
    Si no hay detección antes -> TTA=0, hit=False
    """
    if time_of_event is None:
        return 0.0, False
    if not window_times or not clip_probs:
        return 0.0, False

    times = np.asarray(window_times, dtype=np.float32)
    probs = np.asarray(clip_probs, dtype=np.float32)

    mask = (probs >= thr) & (times <= time_of_event)
    if not mask.any():
        return 0.0, False

    t_detect = float(times[mask].min())
    tta = max(float(time_of_event) - t_detect, 0.0)
    return tta, True

def find_threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.8,
) -> float:
    y_true = (y_true > 0).astype(np.int32)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.5

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    tp = np.cumsum(y_sorted)
    recall = tp / max(1, n_pos)

    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return 0.5
    return float(scores_sorted[idx[0]])


# ==========================================================
# 3) MODELO (idéntico al TRAIN K-tokens)
# ==========================================================

class DepthPatchTokenizer(nn.Module):
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
        self.tubelet_size = int(tubelet_size)
        self.patch_size = int(patch_size)
        self.frames_per_clip = int(frames_per_clip)
        self.crop_size = int(crop_size)

        self.steps = self.frames_per_clip // self.tubelet_size
        self.grid = self.crop_size // self.patch_size
        self.n_patches = self.grid ** 2

        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
            stride=(self.tubelet_size, self.patch_size, self.patch_size),
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

        x = self.patch_embed(depth_btchw)  # [B,hidden,S,grid,grid]
        B2, C2, S, Gh, Gw = x.shape
        assert S == self.steps and Gh == self.grid and Gw == self.grid, \
            f"Depth tokenizer mismatch: got {x.shape}, expected S={self.steps} grid={self.grid}"

        x = x.permute(0, 2, 3, 4, 1).contiguous()          # [B,S,grid,grid,C2]
        x = x.view(B2, self.steps, self.n_patches, C2)     # [B,S,P,C2]
        x = self.proj(x)                                   # [B,S,P,D_depth]
        return x

class DenseRGBDepthFusion(nn.Module):
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
    def __init__(self, dim: int, hidden: int = 256, n_heads: int = 1):
        super().__init__()
        self.n_heads = int(n_heads)
        self.dim = int(dim)

        if self.n_heads == 1:
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.norm = nn.LayerNorm(dim)
            head_hidden = max(hidden // self.n_heads, 64)
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(self.n_heads)
            ])
            self.combine = nn.Linear(dim * self.n_heads, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,S,P,D] -> [B,S,D]
        if self.n_heads == 1:
            scores = self.mlp(z).squeeze(-1)          # [B,S,P]
            w = scores.softmax(dim=-1)                # [B,S,P]
            return (w.unsqueeze(-1) * z).sum(dim=2)   # [B,S,D]
        z_norm = self.norm(z)
        outs = []
        for head in self.heads:
            scores = head(z_norm).squeeze(-1)
            w = scores.softmax(dim=-1)
            outs.append((w.unsqueeze(-1) * z).sum(dim=2))  # [B,S,D]
        return self.combine(torch.cat(outs, dim=-1))        # [B,S,D]

class MultiQuerySpatialPooler(nn.Module):
    """
    K tokens por step.
      in : [B,S,P,D]
      out: [B,S,K,D]
    """
    def __init__(self, dim: int, num_queries: int = 4, attn_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.K = int(num_queries)
        d_attn = int(attn_dim) if attn_dim is not None else self.dim

        self.norm = nn.LayerNorm(self.dim)
        self.q = nn.Parameter(torch.randn(self.K, d_attn) * 0.02)

        self.k_proj = nn.Linear(self.dim, d_attn, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.out = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(float(dropout)),
        )
        self.scale = d_attn ** -0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,S,P,D]
        B, S, P, D = z.shape
        z = self.norm(z)
        k = self.k_proj(z)  # [B,S,P,d_attn]
        v = self.v_proj(z)  # [B,S,P,D]
        q = self.q[None, None, :, :].expand(B, S, self.K, -1)  # [B,S,K,d_attn]

        attn = torch.einsum("bskd,bspd->bskp", q, k) * self.scale  # [B,S,K,P]
        w = attn.softmax(dim=-1)
        g = torch.einsum("bskp,bspd->bskd", w, v)                 # [B,S,K,D]
        return self.out(g)

class VJEPA2DepthKTokensBinary(nn.Module):
    def __init__(
        self,
        hf_repo: str,
        depth_dim: int = 128,
        spatial_heads: int = 1,
        spatial_queries: int = 1,
        k_reduce: str = "max",
        attn_dim: Optional[int] = None,
        unfreeze_blocks: int = 0,
        encoder_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.spatial_queries = int(spatial_queries)
        self.k_reduce = str(k_reduce).lower()
        assert self.spatial_queries >= 1
        assert self.k_reduce in {"max", "mean", "logsumexp"}

        self.vjepa2 = VJEPA2Model.from_pretrained(hf_repo)
        cfg = self.vjepa2.config
        self.embed_dim = int(cfg.hidden_size)
        self.frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))
        self.crop_size = int(getattr(cfg, "crop_size", 256))
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.grid = self.crop_size // self.patch_size
        self.tubelet = int(getattr(cfg, "tubelet_size", 2))
        self.steps = self.frames_per_clip // self.tubelet
        self.n_patches = self.grid ** 2

        if encoder_ckpt:
            sd = torch.load(encoder_ckpt, map_location="cpu")
            if "model_state" in sd:
                sd = sd["model_state"]
            self.vjepa2.load_state_dict(sd, strict=False)

        # Freeze (eval)
        for p in self.vjepa2.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0:
            self._unfreeze_blocks(unfreeze_blocks)

        self.depth_tokenizer = DepthPatchTokenizer(
            depth_dim=depth_dim,
            tubelet_size=self.tubelet,
            patch_size=self.patch_size,
            frames_per_clip=self.frames_per_clip,
            crop_size=self.crop_size,
            hidden_dim=128,
        )

        self.fused_dim = self.embed_dim + int(depth_dim)
        self.fusion = DenseRGBDepthFusion(self.embed_dim, int(depth_dim), self.fused_dim)

        if self.spatial_queries <= 1:
            self.pooler = SpatialPooler(dim=self.fused_dim, hidden=256, n_heads=spatial_heads)
        else:
            self.pooler = MultiQuerySpatialPooler(dim=self.fused_dim, num_queries=self.spatial_queries, attn_dim=attn_dim)

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

    def _unfreeze_blocks(self, n: int):
        enc = getattr(self.vjepa2, "encoder", None)
        blocks = None
        if enc is not None:
            blocks = getattr(enc, "layers", getattr(enc, "layer", None))
        if blocks is None:
            return
        total = len(blocks)
        for i, block in enumerate(blocks):
            if i >= total - n:
                for p in block.parameters():
                    p.requires_grad = True

    def _rgb_tokens_spatio_temporal(self, pv: torch.Tensor) -> torch.Tensor:
        out = self.vjepa2(pixel_values_videos=pv, output_hidden_states=False)
        tokens = out.last_hidden_state  # [B,L,D] (puede incluir CLS)
        B, L, D = tokens.shape
        expected_L = self.steps * self.n_patches
        if L != expected_L:
            tokens = tokens[:, 1:, :]
            assert tokens.shape[1] == expected_L, f"L mismatch: {tokens.shape[1]} vs {expected_L}"
        return tokens.view(B, self.steps, self.n_patches, D)  # [B,S,P,D]

    def _reduce_k(self, logits_tk: torch.Tensor) -> torch.Tensor:
        # [B,T,K] -> [B,T]
        if self.k_reduce == "max":
            return logits_tk.max(dim=2).values
        if self.k_reduce == "mean":
            return logits_tk.mean(dim=2)
        return torch.logsumexp(logits_tk, dim=2)

    def forward(self, pixel_values_videos: torch.Tensor, depth_videos: torch.Tensor, return_frame_logits: bool = False):
        B = pixel_values_videos.shape[0]

        z_rgb = self._rgb_tokens_spatio_temporal(pixel_values_videos)  # [B,S,P,D_rgb]
        z_depth = self.depth_tokenizer(depth_videos)                   # [B,S,P,D_depth]
        z_fused = self.fusion(z_rgb, z_depth)                          # [B,S,P,D_fused]

        pooled = self.pooler(z_fused)

        if self.spatial_queries <= 1:
            # pooled [B,S,D]
            z_steps = pooled                                           # [B,S,D]
            z_temporal = z_steps.transpose(1, 2)                       # [B,D,S]
            z_temporal = F.interpolate(z_temporal, size=self.frames_per_clip, mode="linear", align_corners=False)
            z_temporal = z_temporal.transpose(1, 2).contiguous()       # [B,T,D]

            T = z_temporal.shape[1]
            mask = torch.ones(B, T, device=z_temporal.device, dtype=torch.bool)
            logits_bt = self.temporal_head(z_temporal, mask=mask).squeeze(-1)  # [B,T]

        else:
            # pooled [B,S,K,D]
            z_sk = pooled                                              # [B,S,K,D]
            B2, S2, K2, D2 = z_sk.shape

            # interp S->T por query (fold K en batch)
            z = z_sk.permute(0, 2, 3, 1).contiguous().view(B2 * K2, D2, S2)    # [B*K,D,S]
            z = F.interpolate(z, size=self.frames_per_clip, mode="linear", align_corners=False)  # [B*K,D,T]
            z = z.view(B2, K2, D2, self.frames_per_clip).permute(0, 3, 1, 2).contiguous()       # [B,T,K,D]

            # temporal head sobre T*K (flatten)
            z_flat = z.view(B2, self.frames_per_clip * K2, D2)               # [B,T*K,D]
            mask = torch.ones(B2, self.frames_per_clip * K2, device=z_flat.device, dtype=torch.bool)
            logits_flat = self.temporal_head(z_flat, mask=mask).squeeze(-1)  # [B,T*K]

            logits_tk = logits_flat.view(B2, self.frames_per_clip, K2)       # [B,T,K]
            logits_bt = self._reduce_k(logits_tk)                             # [B,T]

        if return_frame_logits:
            return logits_bt
        return logits_bt.max(dim=1).values  # anytime hard


# ==========================================================
# 4) EVALUACIÓN SLIDING
# ==========================================================

@torch.no_grad()
def evaluate_split(
    rows: List[Dict[str, Any]],
    model: VJEPA2DepthKTokensBinary,
    processor: AutoVideoProcessor,
    device: torch.device,
    video_map: Dict[str, Path],
    depth_map: Dict[str, Path],
    frames_per_clip: int,
    hist_s: float,
    stride_s: float,
    agg: str,
    top_k: int,
    batch_size: int,
    tta_thr: float,
    depth_max_m: float,
    depth_log_div: float,
    out_csv: Optional[Path] = None,
) -> Dict[str, Any]:

    iterator = tqdm(rows, desc="Eval", unit="vid") if tqdm is not None else rows

    all_scores: List[float] = []
    all_targets: List[int] = []
    all_ids: List[str] = []

    n_missing = 0

    # TTA @ tta_thr
    tta_values: List[float] = []
    n_pos_with_event = 0
    n_pos_hits_before = 0

    # Para TTA@R80
    pos_window_times_list: List[List[float]] = []
    pos_clip_probs_list: List[List[float]] = []
    pos_time_of_event_list: List[float] = []

    pred_rows: List[Dict[str, Any]] = []

    per_video_times: List[float] = []
    durations: List[float] = []

    for r in iterator:
        vid = r["id"]
        target = int(r["target"])
        t_event = r.get("time_of_event", None)

        vpath = video_map.get(vid, None)
        if vpath is None:
            n_missing += 1
            continue
        dpath = depth_map.get(vid, None)

        # timing E2E vídeo (incluye read + sliding + infer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # read video
        try:
            video, _, info = read_video(str(vpath), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            n_missing += 1
            continue
        if video.numel() == 0:
            n_missing += 1
            continue

        Tframes = video.shape[0]
        duration = Tframes / max(fps, 1e-6)
        durations.append(duration)

        # centers
        if duration <= hist_s:
            centers = [0.5 * duration]
        else:
            n_steps = int(np.floor((duration - hist_s) / stride_s)) + 1
            centers = [hist_s + i * stride_s for i in range(max(1, n_steps))]
            if centers[-1] < duration - 0.25:
                centers.append(duration)

        rgb_clips: List[np.ndarray] = []
        depth_clips: List[np.ndarray] = []
        window_times: List[float] = []

        for c in centers:
            end_s = min(duration, c)
            start_s = max(0.0, end_s - hist_s)

            start_idx = int(round(start_s * fps))
            end_idx = int(round(end_s * fps))

            start_idx = max(0, min(start_idx, Tframes - 1))
            end_idx = max(start_idx, min(end_idx, Tframes - 1))

            if start_idx == end_idx:
                idx = np.full((frames_per_clip,), start_idx, dtype=np.int64)
            else:
                idx = np.linspace(start_idx, end_idx, frames_per_clip).astype(np.int64)
                idx = np.clip(idx, 0, Tframes - 1)

            rgb_clip = video[idx].numpy()  # [T,H,W,3]
            rgb_clips.append(rgb_clip)

            if dpath is not None:
                d_clip = load_depth_clip_nearest(dpath, idx, frames_per_clip)
            else:
                d_clip = np.zeros((frames_per_clip, 224, 224, 1), dtype=np.float32)
            depth_clips.append(d_clip)

            window_times.append(end_s)

        if not rgb_clips:
            n_missing += 1
            continue

        # infer mini-batch
        clip_probs: List[float] = []
        model.eval()

        for i in range(0, len(rgb_clips), batch_size):
            batch_rgb = rgb_clips[i:i + batch_size]
            batch_depth = depth_clips[i:i + batch_size]

            inputs = processor(batch_rgb, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            d_np = np.stack(batch_depth, axis=0).astype(np.float32)  # [B,T,H,W,1]
            d_t = torch.from_numpy(d_np).float()[..., 0]             # [B,T,H,W]

            # depth norm (must match train)
            d_t = torch.clamp(d_t, min=0.0, max=float(depth_max_m))
            d_t = torch.log1p(d_t) / float(depth_log_div)
            d_t = d_t.unsqueeze(1).contiguous().to(device, non_blocking=True)  # [B,1,T,H,W]

            logits_clip = model(inputs["pixel_values_videos"], d_t)  # [B] (anytime max interno)
            probs = torch.sigmoid(logits_clip).detach().cpu().numpy().tolist()
            clip_probs.extend(probs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        per_video_times.append(t1 - t0)

        # agregación por vídeo
        cp = np.asarray(clip_probs, dtype=np.float32)
        if cp.size == 0:
            video_score = 0.0
        else:
            if agg == "max":
                video_score = float(cp.max())
            elif agg == "mean":
                video_score = float(cp.mean())
            elif agg == "topk":
                k = min(int(top_k), int(cp.size))
                video_score = float(np.mean(np.sort(cp)[-k:])) if k > 0 else float(cp.max())
            else:
                video_score = float(cp.max())

        all_scores.append(video_score)
        all_targets.append(target)
        all_ids.append(vid)

        pred_rows.append({"id": vid, "target": target, "score": f"{video_score:.6f}"})

        # TTA
        if (target == 1) and (t_event is not None):
            n_pos_with_event += 1
            tta, hit = compute_tta_for_video(window_times, clip_probs, float(t_event), tta_thr)
            if hit:
                n_pos_hits_before += 1
            tta_values.append(tta)

            # para R80
            pos_window_times_list.append(list(window_times))
            pos_clip_probs_list.append(cp.astype(np.float32).tolist())
            pos_time_of_event_list.append(float(t_event))

    # métricas globales
    if not all_scores:
        return {"n_eval": 0, "n_missing": n_missing}

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_score = np.asarray(all_scores, dtype=np.float32)

    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    y_pred = (y_score >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    if tta_values and n_pos_with_event > 0:
        mtta = float(np.mean(tta_values))
        recall_before = n_pos_hits_before / max(1, n_pos_with_event)
    else:
        mtta = float("nan")
        recall_before = 0.0

    if per_video_times:
        avg_t = float(np.mean(per_video_times))
        avg_dur = float(np.mean(durations)) if durations else float("nan")
        xrt = avg_dur / avg_t if (avg_t > 0 and not np.isnan(avg_dur)) else float("nan")
    else:
        avg_t = float("nan")
        xrt = float("nan")

    # guardar CSV
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "target", "score"])
            w.writeheader()
            w.writerows(pred_rows)
        print(f"[CSV] Predicciones guardadas en: {out_csv}")

    # bloque R@80
    thr_r80 = find_threshold_for_recall(y_true, y_score, target_recall=0.8)
    y_pred_r80 = (y_score >= thr_r80).astype(np.int64)

    acc_r80 = accuracy_score(y_true, y_pred_r80)
    prec_r80 = precision_score(y_true, y_pred_r80, zero_division=0)
    rec_r80 = recall_score(y_true, y_pred_r80, zero_division=0)
    f1_r80 = f1_score(y_true, y_pred_r80, zero_division=0)

    # mTTA@R80
    if pos_window_times_list:
        tta_vals_r80: List[float] = []
        hits_before_r80 = 0
        for w_times, c_probs, tev in zip(pos_window_times_list, pos_clip_probs_list, pos_time_of_event_list):
            tta_r, hit_r = compute_tta_for_video(w_times, c_probs, tev, thr_r80)
            if hit_r:
                hits_before_r80 += 1
            tta_vals_r80.append(tta_r)

        mtta_r80 = float(np.mean(tta_vals_r80)) if tta_vals_r80 else float("nan")
        rec_before_r80 = hits_before_r80 / max(1, len(tta_vals_r80))
    else:
        mtta_r80 = float("nan")
        rec_before_r80 = 0.0

    return {
        "n_eval": len(all_scores),
        "n_missing": n_missing,
        "ap": ap,
        "auc": auc,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "cm": [[tn, fp], [fn, tp]],
        "mtta_thr": tta_thr,
        "mtta": mtta,
        "recall_before_event": recall_before,
        "avg_time_per_video": avg_t,
        "xrt": xrt,

        "thr_r80": thr_r80,
        "acc_r80": acc_r80,
        "prec_r80": prec_r80,
        "rec_r80": rec_r80,
        "f1_r80": f1_r80,
        "mtta_r80": mtta_r80,
        "recall_before_event_r80": rec_before_r80,
    }


# ==========================================================
# 5) MAIN
# ==========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--video-root", type=str, required=True)
    ap.add_argument("--depth-root", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="preds_depth_k_tokens_sliding.csv")

    # sliding
    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--stride-s", type=float, default=0.5)

    # agregación
    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean", "topk"])
    ap.add_argument("--top-k", type=int, default=5)

    # infer
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tta-thr", type=float, default=0.5)

    # overrides opcionales (si no, lee del checkpoint)
    ap.add_argument("--depth-max-m", type=float, default=-1.0, help="Override. -1 usa el del checkpoint o 150.")
    ap.add_argument("--depth-log-div", type=float, default=-1.0, help="Override. -1 usa el del checkpoint o 5.")
    ap.add_argument("--spatial-queries", type=int, default=-1, help="Override K. -1 usa el del checkpoint.")
    ap.add_argument("--k-reduce", type=str, default="", help="Override reduce {max,mean,logsumexp}. '' usa el del checkpoint.")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERR] No encuentro checkpoint: {ckpt_path}")
        sys.exit(1)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERR] No encuentro CSV: {csv_path}")
        sys.exit(1)

    video_root = Path(args.video_root)
    depth_root = Path(args.depth_root)

    # load checkpoint
    print(f"[INIT] Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    hf_repo = ckpt_args.get("hf_repo", "facebook/vjepa2-vitl-fpc16-256-ssv2")
    depth_dim = int(ckpt_args.get("depth_dim", 128))
    spatial_heads = int(ckpt_args.get("spatial_heads", 1))
    spatial_queries = int(ckpt_args.get("spatial_queries", 1))
    k_reduce = str(ckpt_args.get("k_reduce", "max")).lower()
    attn_dim_raw = ckpt_args.get("attn_dim", 0)
    attn_dim = None if (attn_dim_raw is None or int(attn_dim_raw) == 0) else int(attn_dim_raw)

    depth_max_m = float(ckpt_args.get("depth_max_m", 150.0))
    depth_log_div = float(ckpt_args.get("depth_log_div", 5.0))

    # overrides CLI
    if args.depth_max_m > 0:
        depth_max_m = float(args.depth_max_m)
    if args.depth_log_div > 0:
        depth_log_div = float(args.depth_log_div)
    if args.spatial_queries > 0:
        spatial_queries = int(args.spatial_queries)
    if args.k_reduce.strip():
        k_reduce = args.k_reduce.strip().lower()

    # model
    model = VJEPA2DepthKTokensBinary(
        hf_repo=hf_repo,
        depth_dim=depth_dim,
        spatial_heads=spatial_heads,
        spatial_queries=spatial_queries,
        k_reduce=k_reduce,
        attn_dim=attn_dim,
        unfreeze_blocks=0,
        encoder_ckpt=None,
    )

    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (ej: {missing[:8]})")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (ej: {unexpected[:8]})")

    # sanity: si falta algo gordo, peta
    critical_prefix = ("fusion", "pooler", "temporal_head", "depth_tokenizer", "vjepa2")
    bad_missing = [k for k in missing if k.startswith(critical_prefix)]
    if bad_missing:
        raise RuntimeError(f"[ERR] Missing CRÍTICO (state_dict incompatible): {bad_missing[:12]}")

    model.to(device).eval()

    model_size_mb = get_model_size_mb(model)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 80)
    print(f"MODELO: VJEPA2+Depth+KTokens | Params: {n_params/1e6:.1f}M | Size: {model_size_mb:.1f} MB")
    print(f"CFG: K={spatial_queries} reduce={k_reduce} depth_dim={depth_dim} depth_norm=clamp{depth_max_m}/logdiv{depth_log_div}")
    print(f"SLIDING: hist={args.hist_s}s stride={args.stride_s}s batch={args.batch_size} agg={args.agg}-{args.top_k}")
    print("=" * 80)

    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    cfg = AutoConfig.from_pretrained(hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))
    print(f"[CFG] frames_per_clip={frames_per_clip}")

    rows = read_csv_with_event(csv_path)
    print(f"[DATA] Vídeos en CSV: {len(rows)}")

    video_map = index_videos(video_root)
    depth_map = index_depth(depth_root)

    out_csv = Path(args.out_csv)

    res = evaluate_split(
        rows=rows,
        model=model,
        processor=processor,
        device=device,
        video_map=video_map,
        depth_map=depth_map,
        frames_per_clip=frames_per_clip,
        hist_s=args.hist_s,
        stride_s=args.stride_s,
        agg=args.agg,
        top_k=args.top_k,
        batch_size=args.batch_size,
        tta_thr=args.tta_thr,
        depth_max_m=depth_max_m,
        depth_log_div=depth_log_div,
        out_csv=out_csv,
    )

    print("\n========== RESULTADOS SLIDING + TTA ==========")
    print(f"Videos evaluados   : {res.get('n_eval', 0)}")
    print(f"Videos missing     : {res.get('n_missing', 0)}")
    print(f"AP_anytime         : {res.get('ap', float('nan')):.6f}")
    print(f"AUC_anytime        : {res.get('auc', float('nan')):.6f}")
    print(f"Acc@0.5            : {res.get('acc', 0.0)*100:.2f}%")
    print(f"Prec@0.5           : {res.get('prec', 0.0):.3f}")
    print(f"Rec@0.5            : {res.get('rec', 0.0):.3f}")
    print(f"F1@0.5             : {res.get('f1', 0.0):.3f}")
    print(f"Confusion [ [TN FP], [FN TP] ] = {res.get('cm')}")
    print(f"mTTA@{res.get('mtta_thr', 0.5):.2f}        : {res.get('mtta', float('nan')):.3f} s")
    print(f"Rec_before_event@{res.get('mtta_thr', 0.5):.2f}: {res.get('recall_before_event', 0.0)*100:.2f}%")
    print(f"Avg infer time/video: {res.get('avg_time_per_video', float('nan'))*1000:.2f} ms")
    print(f"x Real-time         : {res.get('xrt', float('nan')):.3f}x")
    print("==============================================")
    print(f"thr_R80             : {res.get('thr_r80', float('nan')):.3f}")
    print(f"Acc@R80             : {res.get('acc_r80', 0.0)*100:.2f}%")
    print(f"F1@R80              : {res.get('f1_r80', 0.0):.3f} "
          f"(P:{res.get('prec_r80', 0.0):.2f} R:{res.get('rec_r80', 0.0):.2f})")
    print(f"mTTA@R80            : {res.get('mtta_r80', float('nan')):.3f} s")
    print(f"Rec_before_event@R80: {res.get('recall_before_event_r80', 0.0)*100:.2f}%")
    print("==============================================\n")


if __name__ == "__main__":
    main()
