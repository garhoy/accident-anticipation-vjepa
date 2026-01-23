#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_nexar_vjepa_depth_sliding_tta.py

Evaluación REAL-WORLD (sliding window) para:

    V-JEPA 2 (facebook/vjepa2-vitl-fpc16-256-ssv2)
  + DepthAnything3 (mapas de profundidad precomputados .npz)
  + Temporal head frame-wise (VJEPA2DepthTransformerBinary)

Métricas:
  - AP_anytime, AUC_anytime (clasificación binaria por vídeo)
  - Acc, Prec, Rec, F1 @ 0.5
  - mTTA@thr (mean Time-To-Accident, usando ventanas deslizantes)
  - Recall_before_event@thr (proporción de positivos detectados ANTES del evento)

Requisitos CSV:
  - Columnas mínimas: id, target
  - Para TTA: columna opcional 'time_of_event' (float, segundos).
    Si no está o es NaN, ese vídeo NO contribuye a TTA.

Uso típico:

  CUDA_VISIBLE_DEVICES=0 python eval_nexar_vjepa_depth_sliding_tta.py \
    --checkpoint checkpoints_vjepa_da3_framewise/best_model_robust.pt \
    --csv "/home/ander/V-JEPA 2/data/metadata/Nexar/val.csv" \
    --video-root "/home/ander/V-JEPA 2/data/raw/Nexar" \
    --depth-root "/home/ander/V-JEPA 2/data/processed/Nexar_DA3_Tensors" \
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

# Importa el factory de heads (TransformerPerFrame)
from models import build_model

from thesis.utils.csv_utils import float_or_none


# ==========================================================
# 1. PARSEO CSV / HELPERS
# ==========================================================

def get_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def read_csv_with_event(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Espera columnas:
      - id (str)
      - target (0/1)
      - time_of_event (float, opcional)
    Ignora filas sin id o sin target parseable.
    """
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
# 2. INDEXADO DE VÍDEOS Y DEPTH
# ==========================================================

def index_videos(video_root: Path) -> Dict[str, Path]:
    """
    Escanea video_root recursivamente y construye:
        id (stem) -> ruta completa del vídeo
    """
    valid_exts = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}
    video_map: Dict[str, Path] = {}
    print(f"[INDEX] Indexando vídeos en {video_root} ...")
    for p in video_root.rglob("*"):
        if p.suffix in valid_exts:
            video_map[p.stem] = p
    print(f"[INDEX] Vídeos indexados: {len(video_map)}")
    return video_map


def index_depth(depth_root: Path) -> Dict[str, Path]:
    """
    Escanea depth_root recursivamente y construye:
        id (stem) -> ruta completa del .npz
    """
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
    Carga depth .npz y alinea a los frame_indices vía nearest neighbor.
    Devuelve [T, H, W, 1] float32.
    Si falla, devuelve zeros.
    """
    try:
        data = np.load(str(npz_path))
        d_stack = data["depth"]       # [N, H, W]
        d_idx = data["frame_idx"]     # [N]
        if d_stack.shape[0] == 0:
            raise RuntimeError("Empty depth stack")

        diffs = np.abs(d_idx[None, :] - frame_indices[:, None])
        nearest = diffs.argmin(axis=1)        # [T]
        clip = d_stack[nearest]               # [T, H, W]
        clip = clip[..., None].astype(np.float32)  # [T, H, W, 1]
        return clip
    except Exception:
        # Fallback: zeros
        return np.zeros((frames_per_clip, 224, 224, 1), dtype=np.float32)


# ==========================================================
# 3. MODELO (IDÉNTICO AL TRAIN)
# ==========================================================
# ==========================================================
# 3. MODELO (IDÉNTICO AL TRAIN: DepthPatchTokenizer + Fusión densa + SpatialPooler)
# ==========================================================

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

    def _unfreeze_blocks(self, n: int):
        blocks = getattr(self.vjepa2.encoder, "layers",
                         getattr(self.vjepa2.encoder, "layer", None))
        if blocks:
            total = len(blocks)
            for i, block in enumerate(blocks):
                if i >= total - n:
                    for p in block.parameters():
                        p.requires_grad = True

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

# ==========================================================
# 4. EVALUACIÓN SLIDING + TTA
# ==========================================================

def compute_tta_for_video(
    window_times: List[float],
    clip_probs: List[float],
    time_of_event: float,
    thr: float,
) -> Tuple[float, bool]:
    """
    Devuelve:
      - tta (>=0) en segundos
      - hit_before_event (bool): True si hubo detección antes del evento

    Definición:
      t_detect = min t_window s.t. prob(t_window) >= thr y t_window <= time_of_event
      TTA     = max(time_of_event - t_detect, 0)
      Si no hay ventana por encima de umbral ANTES del evento -> TTA = 0, hit=False.
    """
    if time_of_event is None:
        return 0.0, False

    if not window_times or not clip_probs:
        return 0.0, False

    times = np.asarray(window_times, dtype=np.float32)
    probs = np.asarray(clip_probs, dtype=np.float32)

    mask = probs >= thr
    mask = mask & (times <= time_of_event)

    if not mask.any():
        # No anticipa antes del evento
        return 0.0, False

    t_detect = float(times[mask].min())
    tta = max(time_of_event - t_detect, 0.0)
    return tta, True




def find_threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.8,
) -> float:
    """
    Devuelve el umbral mínimo tal que recall >= target_recall.
    Si no se alcanza, devuelve 0.5 como fallback.
    """
    y_true = (y_true > 0).astype(np.int32)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.5

    # Ordenamos por score descendente
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    tp = np.cumsum(y_sorted)
    recall = tp / max(1, n_pos)

    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return 0.5

    # Primer score donde alcanzas la recall objetivo
    return float(scores_sorted[idx[0]])


def evaluate_split(
    rows: List[Dict[str, Any]],
    model: VJEPA2DepthTransformerBinary,
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
    out_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evalúa un conjunto de vídeos con sliding window.

    Devuelve métricas estándar @0.5 y además bloque @R80:
      - thr_r80
      - Acc/Prec/Rec/F1@R80
      - mTTA@R80
      - Recall_before_event@R80
    """
    if tqdm is not None:
        iterator = tqdm(rows, desc="Eval", unit="vid")
    else:
        iterator = rows

    all_scores: List[float] = []
    all_targets: List[int] = []
    all_ids: List[str] = []

    n_missing = 0

    # TTA @ tta_thr (típico 0.5)
    tta_values: List[float] = []
    n_pos_with_event = 0
    n_pos_hits_before = 0

    # Para TTA@R80
    pos_window_times_list: List[List[float]] = []
    pos_clip_probs_list: List[List[float]] = []
    pos_time_of_event_list: List[float] = []

    # Opcional: salvar CSV
    pred_rows: List[Dict[str, Any]] = []

    # Stats de tiempo
    per_video_times: List[float] = []
    durations: List[float] = []

    for r in iterator:
        vid = r["id"]
        target = int(r["target"])
        t_event = r.get("time_of_event", None)

        # Localizar vídeo
        vpath = video_map.get(vid, None)
        if vpath is None:
            n_missing += 1
            continue

        dpath = depth_map.get(vid, None)

        # ==== START TIMING END-TO-END (read_video + sliding + modelo) ====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # ================================================================

        # Leer vídeo
        try:
            video, _, info = read_video(str(vpath), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            n_missing += 1
            continue

        if video.numel() == 0:
            n_missing += 1
            continue

        T = video.shape[0]
        duration = T / max(fps, 1e-6)
        durations.append(duration)

        # ---- Sliding windows (centers en segundos) ----
        centers: List[float] = []
        if duration <= hist_s:
            centers = [0.5 * duration]
        else:
            n_steps = int(np.floor((duration - hist_s) / stride_s)) + 1
            centers = [hist_s + i * stride_s for i in range(max(1, n_steps))]
            # Asegurar cubrir el final del vídeo
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

            start_idx = max(0, min(start_idx, T - 1))
            end_idx = max(start_idx, min(end_idx, T - 1))

            if start_idx == end_idx:
                idx = np.full((frames_per_clip,), start_idx, dtype=np.int64)
            else:
                idx = np.linspace(start_idx, end_idx, frames_per_clip).astype(np.int64)
                idx = np.clip(idx, 0, T - 1)

            rgb_clip = video[idx].numpy()  # [T,H,W,3]
            rgb_clips.append(rgb_clip)

            if dpath is not None:
                d_clip = load_depth_clip_nearest(dpath, idx, frames_per_clip)
            else:
                d_clip = np.zeros((frames_per_clip, 224, 224, 1), dtype=np.float32)
            depth_clips.append(d_clip)

            # Usamos end_s como "tiempo de decisión" de la ventana
            window_times.append(end_s)

        # No hay ventanas (raro, pero por robustez)
        if not rgb_clips:
            n_missing += 1
            continue

        # ---- Inferencia mini-batch ----
        clip_probs: List[float] = []

        model.eval()
        with torch.no_grad():
            for i in range(0, len(rgb_clips), batch_size):
                batch_rgb = rgb_clips[i : i + batch_size]
                batch_depth = depth_clips[i : i + batch_size]

                inputs = processor(batch_rgb, return_tensors="pt")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                d_np = np.stack(batch_depth, axis=0)        # [B,T,H,W,1]
                d_t = torch.from_numpy(d_np).float()       # [B,T,H,W,1]

                # Quitamos canal dummy -> [B,T,H,W]
                d_t = d_t[..., 0]

                # 1) Clamp físico
                d_t = torch.clamp(d_t, min=0.0, max=150.0)

                # 2) Log-escala + reescalado EXACTAMENTE igual que en train
                d_t = torch.log1p(d_t) / 5.0

                # 3) Añadimos canal -> [B,1,T,H,W]
                d_t = d_t.unsqueeze(1).contiguous()

                d_t = d_t.to(device, non_blocking=True)

                logits = model(
                    pixel_values_videos=inputs["pixel_values_videos"],
                    depth_videos=d_t,
                )  # [B] — ya es max_t(logits_frame)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                clip_probs.extend(probs)

        # ==== STOP TIMING END-TO-END ====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        per_video_times.append(t1 - t0)
        # ===================================================================

        # ---- Agregación por vídeo ----
        if not clip_probs:
            video_score = 0.0
        else:
            cp = np.asarray(clip_probs, dtype=np.float32)
            if agg == "max":
                video_score = float(cp.max())
            elif agg == "mean":
                video_score = float(cp.mean())
            elif agg == "topk":
                k = min(top_k, len(cp))
                video_score = float(np.mean(np.sort(cp)[-k:])) if k > 0 else 0.0
            else:
                video_score = float(cp.max())

        all_scores.append(video_score)
        all_targets.append(target)
        all_ids.append(vid)

        pred_rows.append(
            {"id": vid, "target": target, "score": f"{video_score:.6f}"}
        )

        # ---- TTA (solo vídeos positivos con time_of_event válido) ----
        if (target == 1) and (t_event is not None):
            n_pos_with_event += 1
            tta, hit = compute_tta_for_video(
                window_times=window_times,
                clip_probs=clip_probs,
                time_of_event=float(t_event),
                thr=tta_thr,
            )
            if hit:
                n_pos_hits_before += 1
            # TTA definido también para fallos (0.0)
            tta_values.append(tta)

            # Guardar para TTA@R80
            pos_window_times_list.append(window_times)
            pos_clip_probs_list.append(list(cp))  # cp es np.array de clip_probs
            pos_time_of_event_list.append(float(t_event))

    # ==================== MÉTRICAS GLOBALES ====================
    if not all_scores:
        return {
            "n_eval": 0,
            "n_missing": n_missing,
        }

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_score = np.asarray(all_scores, dtype=np.float32)

    # AP / AUC
    ap = average_precision_score(y_true, y_score)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = float("nan")

    # Binarización @ 0.5
    y_pred = (y_score >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Matriz de confusión
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    # TTA @ tta_thr
    if tta_values and n_pos_with_event > 0:
        mtta = float(np.mean(tta_values))
        recall_before = n_pos_hits_before / max(1, n_pos_with_event)
    else:
        mtta = float("nan")
        recall_before = 0.0

    # Tiempo medio vídeo / xRT
    if per_video_times:
        avg_t = float(np.mean(per_video_times))    # tiempo medio de procesado por vídeo (s)
        avg_dur = float(np.mean(durations)) if durations else float("nan")  # duración media del vídeo (s)
        xrt = avg_dur / avg_t if (avg_t > 0 and not np.isnan(avg_dur)) else float("nan")
    else:
        avg_t = float("nan")
        xrt = float("nan")

    # Guardar CSV si procede
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "target", "score"])
            w.writeheader()
            w.writerows(pred_rows)
        print(f"[CSV] Predicciones guardadas en: {out_csv}")

    # ---------- Bloque R@80 ----------
    thr_r80 = find_threshold_for_recall(y_true, y_score, target_recall=0.8)

    y_pred_r80 = (y_score >= thr_r80).astype(np.int64)
    acc_r80 = accuracy_score(y_true, y_pred_r80)
    prec_r80 = precision_score(y_true, y_pred_r80, zero_division=0)
    rec_r80 = recall_score(y_true, y_pred_r80, zero_division=0)
    f1_r80 = f1_score(y_true, y_pred_r80, zero_division=0)

    # mTTA@R80 y Recall_before_event@R80
    if pos_window_times_list and len(pos_window_times_list) == len(pos_time_of_event_list):
        tta_vals_r80: List[float] = []
        hits_before_r80 = 0
        for w_times, c_probs, tev in zip(
            pos_window_times_list, pos_clip_probs_list, pos_time_of_event_list
        ):
            tta_r, hit_r = compute_tta_for_video(
                window_times=w_times,
                clip_probs=c_probs,
                time_of_event=tev,
                thr=thr_r80,
            )
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

        # Métricas @R80
        "thr_r80": thr_r80,
        "acc_r80": acc_r80,
        "prec_r80": prec_r80,
        "rec_r80": rec_r80,
        "f1_r80": f1_r80,
        "mtta_r80": mtta_r80,
        "recall_before_event_r80": rec_before_r80,
    }


# ==========================================================
# 5. MAIN
# ==========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Checkpoint entrenado (best_model_robust.pt)")
    ap.add_argument("--csv", type=str, required=True,
                    help="CSV con columnas id,target[,time_of_event]")
    ap.add_argument("--video-root", type=str, required=True,
                    help="Raíz de los vídeos (se indexa recursivamente).")
    ap.add_argument("--depth-root", type=str, required=True,
                    help="Raíz de los .npz de depth.")
    ap.add_argument("--out-csv", type=str, default="preds_nexar_depth_sliding.csv",
                    help="Ruta para guardar preds por vídeo.")

    # Sliding window
    ap.add_argument("--hist-s", type=float, default=5.0,
                    help="Duración de ventana (segundos).")
    ap.add_argument("--stride-s", type=float, default=0.5,
                    help="Stride temporal (segundos).")

    # Agregación
    ap.add_argument("--agg", type=str, default="max",
                    choices=["max", "mean", "topk"],
                    help="Agregación de scores de ventanas -> score de vídeo.")
    ap.add_argument("--top-k", type=int, default=5,
                    help="k para 'topk'.")

    # Inferencia
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Clips por batch en inferencia.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda o cpu.")
    ap.add_argument("--tta-thr", type=float, default=0.5,
                    help="Umbral de probabilidad para TTA (anticipación).")

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

    # ---------- Carga checkpoint ----------
    print(f"[INIT] Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    ckpt_args = ckpt.get("args", {})
    hf_repo = ckpt_args.get("hf_repo", "facebook/vjepa2-vitl-fpc16-256-ssv2")
    depth_dim = ckpt_args.get("depth_dim", 128)
    spatial_heads = ckpt_args.get("spatial_heads", 1)

    # Instanciar modelo con misma config que el train
    model = VJEPA2DepthTransformerBinary(
        hf_repo=hf_repo,
        depth_dim=depth_dim,
        spatial_heads=spatial_heads,
        unfreeze_blocks=0,
        encoder_ckpt=None,
    )

    
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] Pesos faltantes al cargar state_dict: {len(missing)}")
    if unexpected:
        print(f"[WARN] Claves inesperadas al cargar state_dict: {len(unexpected)}")

    bad_missing = [k for k in missing if k.startswith(("depth_net", "temporal_head", "vjepa2"))]
    bad_unexpected = [k for k in unexpected if k.startswith(("depth_net", "temporal_head", "vjepa2"))]

    print("[DBG] Missing ejemplo:", missing[:10])
    print("[DBG] Unexpected ejemplo:", unexpected[:10])

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Hay claves críticas discrepantes:\n"
            f"missing={bad_missing[:10]}\n"
            f"unexpected={bad_unexpected[:10]}"
        )

    model.to(device)
    model.eval()

    # Stats del modelo (peso y parámetros)
    model_size_mb = get_model_size_mb(model)
    n_params = sum(p.numel() for p in model.parameters())

    print("\n" + "="*60)
    print(f"MODELO: VJEPA2+Depth(TransformerPerFrame) | Params: {n_params/1e6:.1f}M | Size: {model_size_mb:.1f} MB")
    print(f"CONFIG: hist={args.hist_s}s | stride={args.stride_s}s | batch={args.batch_size} | agg={args.agg}-{args.top_k}")
    print("="*60)

    # Processor + frames_per_clip
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    cfg = AutoConfig.from_pretrained(hf_repo)
    frames_per_clip = getattr(cfg, "frames_per_clip", 16)
    print(f"[CFG] frames_per_clip={frames_per_clip} | hist_s={args.hist_s} | stride_s={args.stride_s}")

    # Data
    rows = read_csv_with_event(csv_path)
    print(f"[DATA] Vídeos en CSV: {len(rows)}")

    video_map = index_videos(video_root)
    depth_map = index_depth(depth_root)

    # Eval
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
    print("==============================================")

if __name__ == "__main__":
    main()
