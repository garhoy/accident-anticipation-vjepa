#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dad_tfdepth_robust_sliding_tta.py

Eval (sliding-window + TTA) para el dataset DAD, siguiendo el estilo de:
  examples/Depth_VJEPA/eval_nexar_mid_fusion_sliding_tta.py

Pensado para checkpoints del modelo "TFDepth robust v2" (RGB+Depth) como:
  checkpoints_vjepa_da3_tfdepth_robustv2_unfreeze2_hn05_j0-2_pw2p5/best_model_robust.pt

IMPORTANTE (DAD IDs y rutas):
  - En los CSVs de DAD (metadata/DAD/extractor_csv/*.csv) el campo id es del tipo:
      positive/000001   o   negative/000123
  - En los vídeos, la estructura es:
      .../videos/training/{positive,negative}/000001.mp4
      .../videos/testing/{positive,negative}/000456.mp4
  - Este script soporta pasar --video-root como:
      (A) .../videos               (auto-resuelve training/testing según el nombre del CSV o --prefer-split)
      (B) .../videos/training      (ids coinciden directo: positive/000001)
      (C) .../videos/testing

Depth precomputada (recomendado):
  Genera depth con el script genérico:
    CUDA_VISIBLE_DEVICES=0 python examples/Depth_VJEPA/preprocess_depth_da3.py \
      --video-root "/home/ander/V-JEPA 2/data/raw/DAD/videos" \
      --out-root   "/home/ander/BADAS-Open/data/processed/DAD_DA3_Tensors" \
      --target-fps 5 --save-res 224 --batch-size 8

  Y evalúa apuntando a las mismas raíces (misma estructura relativa):
    CUDA_VISIBLE_DEVICES=0 python examples/Depth_VJEPA/eval_dad_tfdepth_robust_sliding_tta.py \
      --checkpoint "/home/ander/BADAS-Open/checkpoints_vjepa_da3_tfdepth_robustv2_unfreeze2_hn05_j0-2_pw2p5/best_model_robust.pt" \
      --csv        "/home/ander/V-JEPA 2/data/metadata/DAD/extractor_csv/testing.csv" \
      --video-root "/home/ander/V-JEPA 2/data/raw/DAD/videos" \
      --depth-root "/home/ander/BADAS-Open/data/processed/DAD_DA3_Tensors" \
      --hist-s 5.0 --stride-s 0.5 --agg max --batch-size 16 \
      --tta-thr 0.5 --target-recall 0.8 --prefer-split auto
"""

import os
import sys
import csv
import time
import argparse
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from transformers import AutoConfig, AutoVideoProcessor

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


# -------------------------
# PATH: asegurar repo root en sys.path (para imports del train script)
# -------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # .../BADAS-Open
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis.utils.csv_utils import float_or_none


# ==========================================================
# 1) HELPERS CSV / STATS
# ==========================================================


def get_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for p in model.parameters():
        param_size += p.nelement() * p.element_size()
    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.nelement() * b.element_size()
    return (param_size + buffer_size) / (1024**2)


def read_csv_with_event(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = str(r.get("id", "")).strip()
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
# 2) INDEX VIDEO / DEPTH (DAD-friendly: IDs con subcarpetas)
# ==========================================================

VALID_VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}
TIME_EPS = 1e-6


def _posix_no_suffix(p: Path) -> str:
    return p.as_posix().rsplit(".", 1)[0] if p.suffix else p.as_posix()


@dataclass
class MultiMap:
    """
    Mapa: key -> [paths...] para resolver colisiones (p.ej., training vs testing).
    """
    key_to_paths: Dict[str, List[Path]]

    def get_unique(self, key: str, prefer_prefix: Optional[str] = None) -> Optional[Path]:
        paths = self.key_to_paths.get(key)
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]

        if prefer_prefix:
            prefer_prefix = prefer_prefix.strip("/").lower()
            for p in paths:
                parts = p.as_posix().lower().split("/")
                if prefer_prefix in parts:
                    return p

        return paths[0]

    def n_keys(self) -> int:
        return len(self.key_to_paths)

    def n_collisions(self) -> int:
        return sum(1 for v in self.key_to_paths.values() if len(v) > 1)


def index_videos_dad(video_root: Path) -> MultiMap:
    """
    Indexa vídeos y crea claves:
      - full_key: relativo a video_root, sin extensión (p.ej. training/positive/000001)
      - short_key: si full_key empieza por training/ o testing/, también añade sin ese prefijo
                  (p.ej. positive/000001) -> es lo que suele venir en el CSV.
    """
    print(f"[INDEX] Indexando vídeos en {video_root} ...")
    key_to_paths: Dict[str, List[Path]] = {}

    for p in video_root.rglob("*"):
        if p.suffix not in VALID_VIDEO_EXTS:
            continue
        try:
            rel = p.relative_to(video_root)
        except Exception:
            continue

        full_key = _posix_no_suffix(rel)
        key_to_paths.setdefault(full_key, []).append(p)

        parts = rel.as_posix().split("/")
        if parts and parts[0] in {"training", "testing"} and len(parts) >= 2:
            short_key = _posix_no_suffix(Path(*parts[1:]))
            key_to_paths.setdefault(short_key, []).append(p)

    mm = MultiMap(key_to_paths=key_to_paths)
    print(f"[INDEX] Vídeos indexados: keys={mm.n_keys()} | collisions={mm.n_collisions()}")
    return mm


def index_depth_dad(depth_root: Path) -> MultiMap:
    print(f"[INDEX] Indexando depth npz en {depth_root} ...")
    key_to_paths: Dict[str, List[Path]] = {}
    if not depth_root.exists():
        print("[WARN] depth-root no existe.")
        return MultiMap(key_to_paths=key_to_paths)

    for p in depth_root.rglob("*.npz"):
        try:
            rel = p.relative_to(depth_root)
        except Exception:
            continue
        full_key = _posix_no_suffix(rel)
        key_to_paths.setdefault(full_key, []).append(p)

        parts = rel.as_posix().split("/")
        if parts and parts[0] in {"training", "testing"} and len(parts) >= 2:
            short_key = _posix_no_suffix(Path(*parts[1:]))
            key_to_paths.setdefault(short_key, []).append(p)

    mm = MultiMap(key_to_paths=key_to_paths)
    print(f"[INDEX] Depth indexados: keys={mm.n_keys()} | collisions={mm.n_collisions()}")
    return mm


def infer_prefer_split(csv_path: Path, prefer_split_arg: str) -> Optional[str]:
    """
    Decide si preferimos "training" o "testing" cuando hay colisión.
    """
    prefer_split_arg = (prefer_split_arg or "auto").lower()
    if prefer_split_arg in {"training", "testing"}:
        return prefer_split_arg
    if prefer_split_arg in {"none", "no", "null"}:
        return None

    name = csv_path.name.lower()
    if "testing" in name or "test" in name:
        return "testing"
    if "training" in name or "train" in name or "val" in name:
        return "training"
    return None


def load_depth_clip_nearest(
    npz_path: Path,
    frame_indices: np.ndarray,
    frames_per_clip: int,
    rgb_fps: float,
) -> np.ndarray:
    """
    Carga depth .npz y alinea a los frame_indices vía nearest neighbor.
    Devuelve [T, H, W, 1] float32. Fallback zeros.
    """
    try:
        data = np.load(str(npz_path))
        d_stack = data["depth"]       # [N, H, W]
        # Preferimos alineación por tiempo si está disponible (más robusto).
        if "t_sec" in data:
            d_t = data["t_sec"].astype(np.float32)  # [N]
            d_t_is_seconds = True
        else:
            d_idx = data["frame_idx"].astype(np.float32)  # [N]
            d_fps = float(data["fps"]) if "fps" in data else float("nan")
            if np.isfinite(d_fps) and d_fps > 0:
                d_t = d_idx / d_fps
                d_t_is_seconds = True
            else:
                # fallback: unidades de "índice de frame" (NO segundos)
                d_t = d_idx
                d_t_is_seconds = False
        if d_stack.shape[0] == 0:
            raise RuntimeError("Empty depth stack")

        if d_t_is_seconds:
            if not (np.isfinite(float(rgb_fps)) and float(rgb_fps) > 0):
                raise RuntimeError("rgb_fps inválido para alineación por tiempo.")
            rgb_t = frame_indices.astype(np.float32) / float(rgb_fps)
        else:
            rgb_t = frame_indices.astype(np.float32)

        diffs = np.abs(d_t[None, :] - rgb_t[:, None])
        nearest = diffs.argmin(axis=1)        # [T]
        clip = d_stack[nearest]               # [T, H, W]
        clip = clip[..., None].astype(np.float32)
        return clip
    except Exception:
        return np.zeros((frames_per_clip, 224, 224, 1), dtype=np.float32)


# ==========================================================
# 3) TTA / THR
# ==========================================================

def compute_tta_for_video(
    window_times: List[float],
    clip_probs: List[float],
    time_of_event: float,
    thr: float,
) -> Tuple[float, bool]:
    if time_of_event is None:
        return 0.0, False
    if not window_times or not clip_probs:
        return 0.0, False

    times = np.asarray(window_times, dtype=np.float32)
    probs = np.asarray(clip_probs, dtype=np.float32)

    # Estricto: evita contar detecciones exactamente en el instante del evento como "pre-event".
    mask = (probs >= thr) & (times < (time_of_event - TIME_EPS))
    if not mask.any():
        return 0.0, False

    t_detect = float(times[mask].min())
    tta = max(time_of_event - t_detect, 0.0)
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
# 4) IMPORT DINÁMICO DEL MODELO (desde train script)
# ==========================================================

def import_train_module(train_script: Path):
    spec = importlib.util.spec_from_file_location("train_depth_mod", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No puedo importar: {train_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_model_class(mod, forced_class: Optional[str] = None):
    """
    Elige clase nn.Module que tenga forward(pixel_values_videos, depth_videos, ...)
    """
    if forced_class:
        if not hasattr(mod, forced_class):
            raise RuntimeError(f"--model-class {forced_class} no existe en {mod.__file__}")
        cls = getattr(mod, forced_class)
        return cls

    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, nn.Module):
            continue
        if obj is nn.Module:
            continue

        if name in {"DepthPatchTokenizer", "DenseRGBDepthFusion", "SpatialPooler", "Depth3DPerFrame"}:
            continue

        try:
            sig = inspect.signature(obj.forward)
            params = list(sig.parameters.keys())
        except Exception:
            continue

        has_rgb = any("pixel_values" in p for p in params)
        has_depth = any("depth" in p for p in params)
        if not (has_rgb and has_depth):
            continue

        score = 0
        low = name.lower()
        if "vjepa" in low:
            score += 3
        if "transform" in low:
            score += 2
        if "binary" in low:
            score += 1
        candidates.append((score, name, obj))

    if not candidates:
        raise RuntimeError(
            "No he encontrado una clase modelo compatible en el train script.\n"
            "Pásame --model-class <NombreClase> o revisa que el modelo tenga forward(pixel_values_videos, depth_videos, ...)."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    print(f"[MODEL_PICK] Using class: {best[1]} (score={best[0]})")
    return best[2]


def instantiate_model_from_ckpt(ModelCls, ckpt_args: Dict[str, Any]):
    """
    Instancia el modelo con args del checkpoint filtrando por la firma real del __init__.
    Funciona con el modelo robust (VJEPA2DepthTransformerBinary) y similares.
    """
    accepted = set(inspect.signature(ModelCls.__init__).parameters.keys())
    accepted.discard("self")

    hf_repo = ckpt_args.get("hf_repo", "facebook/vjepa2-vitl-fpc16-256-ssv2")
    depth_dim = int(ckpt_args.get("depth_dim", 128))
    unfreeze_blocks = int(ckpt_args.get("unfreeze_blocks", 0))
    encoder_ckpt = ckpt_args.get("encoder_ckpt", None)

    kwargs = {
        "hf_repo": hf_repo,
        "depth_dim": depth_dim,
        "unfreeze_blocks": unfreeze_blocks,
        "encoder_ckpt": encoder_ckpt,
    }

    # sinónimos comunes
    if "unfreeze_blocks" not in accepted and "unfreeze" in accepted:
        kwargs["unfreeze"] = kwargs.pop("unfreeze_blocks")
    if "encoder_ckpt" not in accepted and "encoder_checkpoint" in accepted:
        kwargs["encoder_checkpoint"] = kwargs.pop("encoder_ckpt")

    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    print("[INIT_KWARGS] Using kwargs:", kwargs)
    return ModelCls(**kwargs)


# ==========================================================
# 5) EVAL SLIDING
# ==========================================================

def _compute_centers(duration: float, hist_s: float, stride_s: float) -> List[float]:
    """
    Genera tiempos de fin de ventana (end_s) desde el inicio del vídeo:
      end_s = stride, 2*stride, ..., duration
    La ventana es start=max(0, end-hist), así que al principio son ventanas más cortas.
    """
    duration = float(duration)
    stride_s = float(max(float(stride_s), 1e-6))
    if duration <= 0:
        return []

    if duration <= stride_s + 1e-6:
        return [duration]

    centers = np.arange(stride_s, duration + 1e-6, stride_s, dtype=np.float32).tolist()
    if not centers or abs(float(centers[-1]) - duration) > 1e-3:
        centers.append(duration)
    return [float(c) for c in centers if c > 0.0]


def evaluate_split(
    rows: List[Dict[str, Any]],
    model: nn.Module,
    processor: AutoVideoProcessor,
    device: torch.device,
    video_index: MultiMap,
    depth_index: MultiMap,
    prefer_split: Optional[str],
    frames_per_clip: int,
    hist_s: float,
    stride_s: float,
    agg: str,
    top_k: int,
    batch_size: int,
    tta_thr: float,
    target_recall: float,
    out_csv: Optional[Path] = None,
) -> Dict[str, Any]:

    iterator: Iterable = tqdm(rows, desc="Eval", unit="vid") if tqdm else rows

    # Scores "pre-event" para métricas de anticipation:
    #   - negativos: score en todo el vídeo
    #   - positivos: score sólo con ventanas time < t_event (estricto, con eps)
    #   - positivos sin t_event válido: EXCLUIDOS de estas métricas
    all_scores: List[float] = []
    all_targets: List[int] = []
    all_ids: List[str] = []

    # Scores "anytime" (clasificación por vídeo, puede usar post-evento)
    all_scores_anytime: List[float] = []
    all_targets_anytime: List[int] = []
    pred_rows: List[Dict[str, Any]] = []

    n_missing_video = 0
    n_missing_depth = 0
    n_pos_invalid_event = 0

    # TTA @ tta_thr
    tta_values: List[float] = []
    tta_hit_values: List[float] = []
    n_pos_with_event = 0
    n_pos_hits_before = 0

    # Para TTA@R80
    pos_window_times_list: List[List[float]] = []
    pos_clip_probs_list: List[List[float]] = []
    pos_time_of_event_list: List[float] = []
    pos_pre_event_scores: List[float] = []

    per_video_times: List[float] = []
    durations: List[float] = []

    model.eval()

    for r in iterator:
        sample_id = r["id"]
        target = int(r["target"])
        t_event = r.get("time_of_event", None)

        vpath = video_index.get_unique(sample_id, prefer_prefix=prefer_split)
        if vpath is None:
            # segundo intento: si el CSV viene sin prefijo y la raíz es .../videos, probar split/id
            if prefer_split:
                vpath = video_index.get_unique(f"{prefer_split}/{sample_id}", prefer_prefix=prefer_split)
        if vpath is None:
            n_missing_video += 1
            continue

        dpath = depth_index.get_unique(sample_id, prefer_prefix=prefer_split)
        if dpath is None and prefer_split:
            dpath = depth_index.get_unique(f"{prefer_split}/{sample_id}", prefer_prefix=prefer_split)
        if dpath is None:
            n_missing_depth += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # read video
        try:
            video, _, info = read_video(str(vpath), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            n_missing_video += 1
            continue

        if video.numel() == 0:
            n_missing_video += 1
            continue

        T = video.shape[0]
        duration = T / max(fps, 1e-6)
        durations.append(duration)

        # Sanity: en DAD el time_of_event está en segundos.
        # Si falta o está fuera de rango, lo invalidamos (evita leakage/TTAs basura).
        if target == 1:
            if t_event is None:
                n_pos_invalid_event += 1
            else:
                tev = float(t_event)
                if not (0.0 <= tev <= float(duration) + 1e-3):
                    n_pos_invalid_event += 1
                    t_event = None

        centers = _compute_centers(duration=duration, hist_s=hist_s, stride_s=stride_s)
        if not centers:
            n_missing_video += 1
            continue

        rgb_clips: List[np.ndarray] = []
        depth_clips: List[np.ndarray] = []
        window_times: List[float] = []

        for end_s in centers:
            start_s = max(0.0, end_s - hist_s)
            # Anti-leakage: usar floor en el final de ventana y coherencia con filtrado estricto pre-event.
            start_idx = int(np.floor(start_s * fps + TIME_EPS))
            end_idx = int(np.floor(end_s * fps - TIME_EPS))

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
                d_clip = load_depth_clip_nearest(dpath, idx, frames_per_clip, rgb_fps=float(fps))
            else:
                d_clip = np.zeros((frames_per_clip, 224, 224, 1), dtype=np.float32)
            depth_clips.append(d_clip)

            # Tiempo real de decisión coherente con los frames vistos (hasta end_idx inclusive).
            end_time = float(end_idx + 1) / max(float(fps), 1e-6)
            window_times.append(end_time)

        clip_probs: List[float] = []
        with torch.no_grad():
            for i in range(0, len(rgb_clips), batch_size):
                batch_rgb = rgb_clips[i:i + batch_size]
                batch_depth = depth_clips[i:i + batch_size]

                inputs = processor(batch_rgb, return_tensors="pt")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                d_np = np.stack(batch_depth, axis=0)   # [B,T,H,W,1]
                d_t = torch.from_numpy(d_np).float()   # [B,T,H,W,1]
                d_t = d_t[..., 0]                      # [B,T,H,W]

                # Normalización EXACTA a train_nexar_depth_general.py (robust v2)
                d_t = torch.clamp(d_t, 0.0, 150.0)
                d_t = torch.log1p(d_t) / 5.0
                d_t = d_t.unsqueeze(1).contiguous()    # [B,1,T,H,W]
                d_t = d_t.to(device, non_blocking=True)

                logits = model(
                    pixel_values_videos=inputs["pixel_values_videos"],
                    depth_videos=d_t,
                )  # [B] logits clip-level
                probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                clip_probs.extend(probs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        per_video_times.append(t1 - t0)

        # ----------------------------------------------------------
        # SCORE SIN LEAKAGE POST-ACCIDENTE (anticipation válido)
        #
        # - Negativos: agregación sobre todas las ventanas.
        # - Positivos con time_of_event válido: agregación SOLO en ventanas con time < t_event (estricto, con eps).
        # - Positivos sin time_of_event válido: NO se usan para métricas de anticipation.
        # ----------------------------------------------------------
        cp = np.asarray(clip_probs, dtype=np.float32)
        wt = np.asarray(window_times, dtype=np.float32)

        if cp.size == 0:
            pre_score = 0.0
        else:
            if target == 1:
                if t_event is None:
                    cp_use = np.asarray([], dtype=np.float32)
                else:
                    pre = wt < (float(t_event) - TIME_EPS)
                    cp_use = cp[pre] if pre.any() else np.asarray([], dtype=np.float32)
            else:
                cp_use = cp

            if cp_use.size == 0:
                pre_score = 0.0
            else:
                if agg == "max":
                    pre_score = float(cp_use.max())
                elif agg == "mean":
                    pre_score = float(cp_use.mean())
                elif agg == "topk":
                    k = min(top_k, int(cp_use.size))
                    pre_score = float(np.mean(np.sort(cp_use)[-k:])) if k > 0 else 0.0
                else:
                    pre_score = float(cp_use.max())

        video_score = float(pre_score)

        # Score "anytime" (clasificación por vídeo, puede usar ventanas post-evento)
        if cp.size == 0:
            anytime_score = 0.0
        else:
            if agg == "max":
                anytime_score = float(cp.max())
            elif agg == "mean":
                anytime_score = float(cp.mean())
            elif agg == "topk":
                k = min(top_k, int(cp.size))
                anytime_score = float(np.mean(np.sort(cp)[-k:])) if k > 0 else 0.0
            else:
                anytime_score = float(cp.max())

        all_scores_anytime.append(float(anytime_score))
        all_targets_anytime.append(target)

        # Filtrado protocolo: anticipation => (negativos) + (positivos con t_event válido)
        if (target == 0) or (t_event is not None):
            all_scores.append(video_score)
            all_targets.append(target)
            all_ids.append(sample_id)
        pred_rows.append({"id": sample_id, "target": target, "score": f"{video_score:.6f}"})

        # TTA
        if (target == 1) and (t_event is not None):
            n_pos_with_event += 1
            tta, hit = compute_tta_for_video(window_times, clip_probs, float(t_event), tta_thr)
            if hit:
                n_pos_hits_before += 1
                tta_hit_values.append(float(tta))
            tta_values.append(tta)

            pos_window_times_list.append(window_times)
            pos_clip_probs_list.append(clip_probs)
            pos_time_of_event_list.append(float(t_event))
            pos_pre_event_scores.append(float(pre_score))

    if not all_scores:
        return {
            "n_eval": 0,
            "n_eval_anytime": len(all_scores_anytime),
            "n_missing_video": n_missing_video,
            "n_missing_depth": n_missing_depth,
            "n_pos_invalid_event": n_pos_invalid_event,
        }

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_score = np.asarray(all_scores, dtype=np.float32)

    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    y_true_any = np.asarray(all_targets_anytime, dtype=np.int64)
    y_score_any = np.asarray(all_scores_anytime, dtype=np.float32)
    ap_any = average_precision_score(y_true_any, y_score_any) if len(np.unique(y_true_any)) > 1 else float("nan")
    auc_any = roc_auc_score(y_true_any, y_score_any) if len(np.unique(y_true_any)) > 1 else float("nan")

    # @0.5
    y_pred = (y_score >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    # TTA @ tta_thr
    if tta_values and n_pos_with_event > 0:
        mtta = float(np.mean(tta_values))
        mtta_hits_only = float(np.mean(tta_hit_values)) if tta_hit_values else float("nan")
        recall_before = n_pos_hits_before / max(1, n_pos_with_event)
    else:
        mtta = float("nan")
        mtta_hits_only = float("nan")
        recall_before = 0.0

    # speed
    if per_video_times:
        avg_t = float(np.mean(per_video_times))
        avg_dur = float(np.mean(durations)) if durations else float("nan")
        xrt = avg_dur / avg_t if (avg_t > 0 and not np.isnan(avg_dur)) else float("nan")
    else:
        avg_t = float("nan")
        xrt = float("nan")

    # save CSV
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "target", "score"])
            w.writeheader()
            w.writerows(pred_rows)
        print(f"[CSV] Predicciones guardadas en: {out_csv}")

    # R@target_recall
    thr_r80 = find_threshold_for_recall(y_true, y_score, target_recall=target_recall)

    y_pred_r80 = (y_score >= thr_r80).astype(np.int64)
    acc_r80 = accuracy_score(y_true, y_pred_r80)
    prec_r80 = precision_score(y_true, y_pred_r80, zero_division=0)
    rec_r80 = recall_score(y_true, y_pred_r80, zero_division=0)
    f1_r80 = f1_score(y_true, y_pred_r80, zero_division=0)

    # TTA@R80
    if pos_window_times_list:
        tta_vals_r80: List[float] = []
        tta_hit_vals_r80: List[float] = []
        hits_before_r80 = 0
        for w_times, c_probs, tev in zip(pos_window_times_list, pos_clip_probs_list, pos_time_of_event_list):
            tta_r, hit_r = compute_tta_for_video(w_times, c_probs, tev, thr_r80)
            if hit_r:
                hits_before_r80 += 1
                tta_hit_vals_r80.append(float(tta_r))
            tta_vals_r80.append(tta_r)

        mtta_r80 = float(np.mean(tta_vals_r80)) if tta_vals_r80 else float("nan")
        mtta_r80_hits_only = float(np.mean(tta_hit_vals_r80)) if tta_hit_vals_r80 else float("nan")
        rec_before_r80 = hits_before_r80 / max(1, len(tta_vals_r80))
    else:
        mtta_r80 = float("nan")
        mtta_r80_hits_only = float("nan")
        rec_before_r80 = 0.0

    # mTTA curva (barrido de umbrales): integral normalizada de mean_TTA vs recall_before_event.
    mtta_curve = float("nan")
    max_recall_curve = 0.0
    if pos_window_times_list:
        thresholds = np.linspace(1.0, 0.0, 101, dtype=np.float32).tolist()

        recalls: List[float] = []
        mean_ttas: List[float] = []

        for thr in thresholds:
            hits = 0
            tta_vals: List[float] = []
            for w_times, c_probs, tev in zip(pos_window_times_list, pos_clip_probs_list, pos_time_of_event_list):
                tta_thr_i, hit_thr_i = compute_tta_for_video(w_times, c_probs, tev, thr)
                hits += int(hit_thr_i)
                tta_vals.append(float(tta_thr_i))  # incluye 0 si miss

            recall_thr = hits / max(1, len(pos_window_times_list))
            mean_tta_thr = float(np.mean(tta_vals)) if tta_vals else 0.0
            recalls.append(float(recall_thr))
            mean_ttas.append(float(mean_tta_thr))

        order = np.argsort(np.asarray(recalls, dtype=np.float32))
        r = np.asarray([recalls[i] for i in order], dtype=np.float32)
        t = np.asarray([mean_ttas[i] for i in order], dtype=np.float32)
        max_recall_curve = float(r.max()) if r.size else 0.0
        if r.size >= 2 and max_recall_curve > 0:
            area = float(np.trapz(t, r))
            mtta_curve = area / max_recall_curve

    return {
        "n_eval": len(all_scores),
        "n_eval_anytime": len(all_scores_anytime),
        "n_missing_video": n_missing_video,
        "n_missing_depth": n_missing_depth,
        "n_pos_invalid_event": n_pos_invalid_event,

        "ap": ap,
        "auc": auc,
        "ap_anytime": ap_any,
        "auc_anytime": auc_any,

        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "cm": [[tn, fp], [fn, tp]],

        "mtta_thr": tta_thr,
        "mtta": mtta,
        "mtta_hits_only": mtta_hits_only,
        "recall_before_event": recall_before,

        "avg_time_per_video": avg_t,
        "xrt": xrt,

        "thr_r80": thr_r80,
        "acc_r80": acc_r80,
        "prec_r80": prec_r80,
        "rec_r80": rec_r80,
        "f1_r80": f1_r80,
        "mtta_r80": mtta_r80,
        "mtta_r80_hits_only": mtta_r80_hits_only,
        "recall_before_event_r80": rec_before_r80,

        "mtta_curve": mtta_curve,
        "max_recall_curve": max_recall_curve,
    }


# ==========================================================
# 6) MAIN
# ==========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--video-root", type=str, required=True)
    ap.add_argument("--depth-root", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="preds_dad_tfdepth_robust_sliding.csv")

    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--stride-s", type=float, default=0.5)

    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean", "topk"])
    ap.add_argument("--top-k", type=int, default=5)

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tta-thr", type=float, default=0.5)
    ap.add_argument("--target-recall", type=float, default=0.8)

    ap.add_argument(
        "--prefer-split",
        type=str,
        default="auto",
        choices=["auto", "training", "testing", "none"],
        help="Para resolver colisiones de IDs entre training/testing cuando --video-root es la raíz de /videos.",
    )

    ap.add_argument(
        "--train-script",
        type=str,
        default=str(Path(__file__).parent / "train_nexar_depth_general.py"),
        help="Script que define la clase del modelo (se importa dinámicamente).",
    )
    ap.add_argument(
        "--model-class",
        type=str,
        default=None,
        help="Si el auto-pick falla, pon aquí el nombre exacto de la clase del modelo en el train script.",
    )

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    csv_path = Path(args.csv)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No encuentro checkpoint: {ckpt_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"No encuentro CSV: {csv_path}")

    train_script = Path(args.train_script)
    if not train_script.exists():
        raise FileNotFoundError(f"No encuentro train script: {train_script}")

    print(f"[INIT] Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    hf_repo = ckpt_args.get("hf_repo", "facebook/vjepa2-vitl-fpc16-256-ssv2")

    # Entorno "offline" por defecto (evita intentos de red en entornos restringidos).
    # Si el repo no está en caché local, Transformers lanzará un error claro.
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")

    processor = AutoVideoProcessor.from_pretrained(hf_repo, local_files_only=True)
    cfg = AutoConfig.from_pretrained(hf_repo, local_files_only=True)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))

    # construir modelo desde train script
    train_mod = import_train_module(train_script)
    ModelCls = pick_model_class(train_mod, forced_class=args.model_class)
    model = instantiate_model_from_ckpt(ModelCls, ckpt_args)

    # load state
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    critical_substr = ("film", "adapter", "modulat", "temporal_head", "depth", "vjepa2", "transform")
    bad_missing = [k for k in missing if any(s in k.lower() for s in critical_substr)]
    bad_unexpected = [k for k in unexpected if any(s in k.lower() for s in critical_substr)]
    if bad_missing or bad_unexpected:
        print("[ERR] mismatch crítico al cargar checkpoint.")
        print("missing (first 30):", bad_missing[:30])
        print("unexpected (first 30):", bad_unexpected[:30])
        raise RuntimeError("No voy a evaluar un modelo distinto al entrenado. Arregla clase/args y vuelve a intentar.")

    model.to(device)
    model.eval()

    prefer_split = infer_prefer_split(csv_path, args.prefer_split)

    model_size_mb = get_model_size_mb(model)
    n_params = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 70)
    print(f"MODEL: {ModelCls.__name__} | Params: {n_params/1e6:.1f}M | Size: {model_size_mb:.1f} MB")
    print(f"CFG  : frames_per_clip={frames_per_clip} | hist={args.hist_s}s | stride={args.stride_s}s")
    print(f"EVAL : batch={args.batch_size} | agg={args.agg}-{args.top_k} | tta_thr={args.tta_thr} | R@{args.target_recall}")
    print(f"DAD  : prefer_split={prefer_split}")
    print("=" * 70)

    rows = read_csv_with_event(csv_path)
    print(f"[DATA] Vídeos en CSV: {len(rows)}")

    video_root = Path(args.video_root)
    depth_root = Path(args.depth_root)
    video_index = index_videos_dad(video_root)
    depth_index = index_depth_dad(depth_root)

    if depth_index.n_keys() == 0:
        print("[WARN] No se han encontrado .npz en depth-root. Evaluaré con depth=0 (peor rendimiento).")
        print("[HINT] Para precomputar depth en DAD:")
        print(f"       python {Path(__file__).parent / 'preprocess_depth_da3.py'} --video-root \"{video_root}\" --out-root \"{depth_root}\"")

    res = evaluate_split(
        rows=rows,
        model=model,
        processor=processor,
        device=device,
        video_index=video_index,
        depth_index=depth_index,
        prefer_split=prefer_split,
        frames_per_clip=frames_per_clip,
        hist_s=args.hist_s,
        stride_s=args.stride_s,
        agg=args.agg,
        top_k=args.top_k,
        batch_size=args.batch_size,
        tta_thr=args.tta_thr,
        target_recall=args.target_recall,
        out_csv=Path(args.out_csv),
    )

    print("\n========== RESULTADOS DAD (SLIDING + TTA) ==========")
    print(f"Videos evaluados      : {res.get('n_eval', 0)}")
    print(f"Videos evaluados(any) : {res.get('n_eval_anytime', 0)}")
    print(f"Videos missing(video) : {res.get('n_missing_video', 0)}")
    print(f"Videos missing(depth) : {res.get('n_missing_depth', 0)}")
    print(f"Pos sin t_event válido: {res.get('n_pos_invalid_event', 0)}")
    print(f"AP_pre_event          : {res.get('ap', float('nan')):.6f}")
    print(f"AUC_pre_event         : {res.get('auc', float('nan')):.6f}")
    print(f"AP_anytime(full)      : {res.get('ap_anytime', float('nan')):.6f}")
    print(f"AUC_anytime(full)     : {res.get('auc_anytime', float('nan')):.6f}")
    print(f"Acc@0.5               : {res.get('acc', 0.0)*100:.2f}%")
    print(f"Prec@0.5              : {res.get('prec', 0.0):.3f}")
    print(f"Rec@0.5               : {res.get('rec', 0.0):.3f}")
    print(f"F1@0.5                : {res.get('f1', 0.0):.3f}")
    print(f"Confusion [ [TN FP], [FN TP] ] = {res.get('cm')}")
    print(f"mTTA@{res.get('mtta_thr', 0.5):.2f}           : {res.get('mtta', float('nan')):.3f} s")
    print(f"mTTA_hits@{res.get('mtta_thr', 0.5):.2f}      : {res.get('mtta_hits_only', float('nan')):.3f} s")
    print(f"Rec_before_event@{res.get('mtta_thr', 0.5):.2f}: {res.get('recall_before_event', 0.0)*100:.2f}%")
    print(f"Avg infer time/video   : {res.get('avg_time_per_video', float('nan'))*1000:.2f} ms")
    print(f"x Real-time            : {res.get('xrt', float('nan')):.3f}x")
    print("---------------------------------------------------")
    print(f"thr_R{args.target_recall:.2f}             : {res.get('thr_r80', float('nan')):.3f}")
    print(f"Acc@R{args.target_recall:.2f}             : {res.get('acc_r80', 0.0)*100:.2f}%")
    print(f"F1@R{args.target_recall:.2f}              : {res.get('f1_r80', 0.0):.3f} "
          f"(P:{res.get('prec_r80', 0.0):.2f} R:{res.get('rec_r80', 0.0):.2f})")
    print(f"TTA@R{args.target_recall:.2f}             : {res.get('mtta_r80', float('nan')):.3f} s")
    print(f"TTA_hits@R{args.target_recall:.2f}        : {res.get('mtta_r80_hits_only', float('nan')):.3f} s")
    print(f"Rec_before_event@R{args.target_recall:.2f}: {res.get('recall_before_event_r80', 0.0)*100:.2f}%")
    print(f"mTTA_curve             : {res.get('mtta_curve', float('nan')):.3f} s "
          f"(max_recall={res.get('max_recall_curve', 0.0):.3f})")
    print("===================================================")


if __name__ == "__main__":
    main()
