#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dota_tfdepth_robust_sliding_tta.py

Eval (sliding-window + TTA) para el dataset DoTA, siguiendo el protocolo de anticipación:
  - Ventanas desde el inicio: end_s = stride, 2*stride, ..., duration
  - Score por vídeo "pre-event" (sin leakage post-accidente):
      * Positivos con time_of_event válido: agg(prob) SOLO en ventanas con time < t_event (estricto, con eps)
      * Negativos: agg(prob) en todas las ventanas
      * Positivos sin time_of_event válido: EXCLUIDOS de métricas de anticipation
  - thr_R80 se calcula sobre esos scores pre-event
  - Reporta TTA/mTTA y mTTA_curve (barrido de umbral)

Además, para DoTA "como DoTA" (temporal anomaly detection), reporta métricas a nivel ventana usando
la anotación [anomaly_start_s, anomaly_end_s] de ccd_meta.csv:
  - AP_window(DoTA) / AUC_window(DoTA)

CSV DoTA esperado (metadata/DoTA/extractor_csv/*.csv):
  id,target,time_of_event,time_of_alert
  0RJPQ_97dcs_000387,1,4.100,

Inputs soportados:
  - Vídeos: indexa archivos bajo --video-root y resuelve por filename stem == id.
  - Frames (DoTA_seg): si --video-root no contiene vídeos, intenta modo frames con estructura:
        <frames_root>/<id>/images/000000.jpg
    (pasa --video-root apuntando a la carpeta frames, o usa --frames-root explícito).

Depth precomputada (recomendado):
  CUDA_VISIBLE_DEVICES=0 python BADAS-Open/examples/Depth_VJEPA/preprocess_depth_da3.py \
    --video-root "/path/a/DoTA/videos/test" \
    --out-root   "/home/ander/BADAS-Open/data/processed/DoTA_DA3_Tensors/test" \
    --target-fps 5 --save-res 224 --batch-size 8

Eval:
  CUDA_VISIBLE_DEVICES=0 python BADAS-Open/examples/Depth_VJEPA/eval_dota_tfdepth_robust_sliding_tta.py \
    --checkpoint "/home/ander/BADAS-Open/checkpoints_vjepa_da3_tfdepth_robustv2_unfreeze2_hn05_j0-2_pw2p5/best_model_robust.pt" \
    --csv "/home/ander/V-JEPA 2/data/metadata/DoTA/extractor_csv/test.csv" \
    --video-root "/path/a/DoTA/videos" \
    --depth-root "/home/ander/BADAS-Open/data/processed/DoTA_DA3_Tensors" \
    --prefer-split test --hist-s 5.0 --stride-s 0.5 --agg max --batch-size 16 \
    --tta-thr 0.5 --target-recall 0.8
"""

import os
import sys
import csv
import time
import argparse
import importlib.util
import inspect
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from transformers import AutoConfig, AutoVideoProcessor
from PIL import Image
from collections import OrderedDict

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
            tal = float_or_none(r.get("time_of_alert", ""))
            rows.append({"id": vid, "target": t, "time_of_event": tev, "time_of_alert": tal})
    return rows


# ==========================================================
# 2) INDEX VIDEO / FRAMES / DEPTH (DoTA-friendly)
# ==========================================================

VALID_VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}
SPLIT_DIRS = {"train", "val", "test", "training", "testing"}
TIME_EPS = 1e-6


def split_dota_id(clip_id: str) -> Tuple[str, Optional[str]]:
    """
    DoTA extractor_csv puede traer ids "virtuales" como:
      <base_id>__pre   (negativo: tramo pre-accidente del mismo clip)

    En DoTA_seg/frames normalmente SÓLO existe el directorio <base_id>, por lo que:
      - assets (frames/depth/meta fps) se resuelven con base_id
      - para __pre recortamos duración a t_event(base_id) para no incluir el accidente
    """
    clip_id = str(clip_id).strip()
    if "__" in clip_id:
        base, suffix = clip_id.rsplit("__", 1)
        if (suffix in {"pre", "post"} or suffix.startswith("neg")) and base:
            return base, suffix
    return clip_id, None


def _posix_no_suffix(p: Path) -> str:
    return p.as_posix().rsplit(".", 1)[0] if p.suffix else p.as_posix()


@dataclass
class MultiMap:
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


def index_videos_dota(video_root: Path) -> MultiMap:
    print(f"[INDEX] Indexando vídeos en {video_root} ...")
    key_to_paths_set: Dict[str, set] = {}

    for p in video_root.rglob("*"):
        if p.suffix not in VALID_VIDEO_EXTS:
            continue
        try:
            rel = p.relative_to(video_root)
        except Exception:
            continue

        # key por stem (lo que trae el CSV)
        key_to_paths_set.setdefault(p.stem, set()).add(p)

        # key relativo completo (por si el usuario pasa ids con subpath)
        full_key = _posix_no_suffix(rel)
        key_to_paths_set.setdefault(full_key, set()).add(p)

        # key sin split si hay subcarpeta train/val/test
        parts = rel.as_posix().split("/")
        if parts and parts[0].lower() in SPLIT_DIRS and len(parts) >= 2:
            short_key = _posix_no_suffix(Path(*parts[1:]))
            key_to_paths_set.setdefault(short_key, set()).add(p)

    key_to_paths: Dict[str, List[Path]] = {k: sorted(list(v)) for k, v in key_to_paths_set.items()}
    mm = MultiMap(key_to_paths=key_to_paths)
    print(f"[INDEX] Vídeos indexados: keys={mm.n_keys()} | collisions={mm.n_collisions()}")
    return mm


def index_frames_dota(frames_root: Path) -> MultiMap:
    """
    Espera estructura:
      frames_root/<id>/images/000000.jpg
    """
    print(f"[INDEX] Indexando frames en {frames_root} ...")
    key_to_paths: Dict[str, List[Path]] = {}

    if not frames_root.exists():
        return MultiMap(key_to_paths=key_to_paths)

    for p in frames_root.iterdir():
        if not p.is_dir():
            continue
        images_dir = p / "images"
        if not images_dir.is_dir():
            continue

        # check rápido: al menos un jpg/png
        has_img = any(q.suffix.lower() in {".jpg", ".jpeg", ".png"} for q in images_dir.iterdir())
        if not has_img:
            continue

        key_to_paths.setdefault(p.name, []).append(images_dir)

    mm = MultiMap(key_to_paths=key_to_paths)
    print(f"[INDEX] Clips (frames) indexados: keys={mm.n_keys()} | collisions={mm.n_collisions()}")
    return mm


def index_depth_dota(depth_root: Path) -> MultiMap:
    print(f"[INDEX] Indexando depth npz en {depth_root} ...")
    key_to_paths_set: Dict[str, set] = {}
    if not depth_root.exists():
        print("[WARN] depth-root no existe.")
        return MultiMap(key_to_paths={})

    for p in depth_root.rglob("*.npz"):
        try:
            rel = p.relative_to(depth_root)
        except Exception:
            continue

        key_to_paths_set.setdefault(p.stem, set()).add(p)
        full_key = _posix_no_suffix(rel)
        key_to_paths_set.setdefault(full_key, set()).add(p)

        parts = rel.as_posix().split("/")
        if parts and parts[0].lower() in SPLIT_DIRS and len(parts) >= 2:
            short_key = _posix_no_suffix(Path(*parts[1:]))
            key_to_paths_set.setdefault(short_key, set()).add(p)

    key_to_paths: Dict[str, List[Path]] = {k: sorted(list(v)) for k, v in key_to_paths_set.items()}
    mm = MultiMap(key_to_paths=key_to_paths)
    print(f"[INDEX] Depth indexados: keys={mm.n_keys()} | collisions={mm.n_collisions()}")
    return mm


def infer_prefer_split(csv_path: Path, prefer_split_arg: str) -> Optional[str]:
    prefer_split_arg = (prefer_split_arg or "auto").lower()
    if prefer_split_arg in SPLIT_DIRS:
        return prefer_split_arg
    if prefer_split_arg in {"none", "no", "null"}:
        return None

    name = csv_path.name.lower()
    if "test" in name:
        return "test"
    if "val" in name:
        return "val"
    if "train" in name:
        return "train"
    return None


def load_ccd_meta_map(meta_csv: Path) -> Dict[str, Dict[str, float]]:
    """
    Lee DoTA/ccd_meta.csv y devuelve:
      id -> {
          "num_frames": int,
          "duration_s": float,
          "fps": float,
          "anomaly_start_s": float,
          "anomaly_end_s": float,
      }
    """
    out: Dict[str, Dict[str, float]] = {}
    if meta_csv is None or not meta_csv.exists():
        return out
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = str(r.get("id", "")).strip()
            if not vid:
                continue
            try:
                n_frames = int(float(r.get("num_frames", "0")))
            except Exception:
                n_frames = 0
            try:
                dur = float(r.get("duration_s", "nan"))
            except Exception:
                dur = float("nan")
            try:
                a_s = float(r.get("anomaly_start_s", "nan"))
            except Exception:
                a_s = float("nan")
            try:
                a_e = float(r.get("anomaly_end_s", "nan"))
            except Exception:
                a_e = float("nan")
            fps = float(n_frames) / dur if (n_frames > 0 and dur > 0 and not np.isnan(dur)) else float("nan")
            out[vid] = {
                "num_frames": float(n_frames),
                "duration_s": float(dur),
                "fps": float(fps),
                "anomaly_start_s": float(a_s),
                "anomaly_end_s": float(a_e),
            }
    return out


def _list_frame_paths(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    frames = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    frames.sort(key=lambda p: p.name)
    return frames


def read_frames_as_video(images_dir: Path) -> np.ndarray:
    """
    Devuelve np.uint8 [T,H,W,3] desde frames en disco.
    """
    paths = _list_frame_paths(images_dir)
    if not paths:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)

    arrs: List[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        arrs.append(np.asarray(img, dtype=np.uint8))
    return np.stack(arrs, axis=0)


def _cache_get_frame(
    frame_paths: List[Path],
    idx: int,
    cache: "OrderedDict[int, np.ndarray]",
    cache_max: int = 64,
) -> np.ndarray:
    if idx in cache:
        arr = cache.pop(idx)
        cache[idx] = arr
        return arr

    def _resize_short_side(im: Image.Image, short: int = 256) -> Image.Image:
        w, h = im.size
        if w <= 0 or h <= 0:
            return im
        scale = float(short) / float(min(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return im.resize((nw, nh), Image.BILINEAR)

    img = Image.open(frame_paths[idx]).convert("RGB")
    # Cachea reescalado preservando aspect ratio (similar a preproc típico del processor).
    img = _resize_short_side(img, short=256)
    arr = np.asarray(img, dtype=np.uint8)
    cache[idx] = arr
    if len(cache) > cache_max:
        cache.popitem(last=False)
    return arr


def load_depth_clip_nearest(
    npz_path: Path,
    frame_indices: np.ndarray,
    frames_per_clip: int,
    rgb_fps: float,
) -> np.ndarray:
    try:
        data = np.load(str(npz_path))
        d_stack = data["depth"]       # [N, H, W]
        # Preferimos alineación por tiempo si está disponible (más robusto que frame_idx).
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
    if forced_class:
        if not hasattr(mod, forced_class):
            raise RuntimeError(f"--model-class {forced_class} no existe en {mod.__file__}")
        return getattr(mod, forced_class)

    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, nn.Module) or obj is nn.Module:
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
    thr_r80_override: Optional[float] = None,
    out_csv: Optional[Path] = None,
) -> Dict[str, Any]:

    iterator: Iterable = tqdm(rows, desc="Eval", unit="vid") if tqdm else rows

    # Mapa base_id -> time_of_event (segundos). Útil para recortar __pre sin tocar el CSV.
    base_event_time: Dict[str, float] = {}
    for rr in rows:
        tev = rr.get("time_of_event", None)
        if tev is None:
            continue
        base_id, _ = split_dota_id(rr.get("id", ""))
        try:
            base_event_time[base_id] = float(tev)
        except Exception:
            continue

    all_scores: List[float] = []
    all_targets: List[int] = []
    all_ids: List[str] = []

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

    # DoTA standard-ish: métricas a nivel ventana usando [anomaly_start_s, anomaly_end_s] como positivo.
    window_scores: List[float] = []
    window_labels: List[int] = []

    model.eval()

    for r in iterator:
        sample_id = r["id"]
        asset_id, asset_suffix = split_dota_id(sample_id)
        target = int(r["target"])
        t_event = r.get("time_of_event", None)

        vpath = video_index.get_unique(sample_id, prefer_prefix=prefer_split)
        if vpath is None:
            vpath = video_index.get_unique(asset_id, prefer_prefix=prefer_split)
        if vpath is None and prefer_split:
            vpath = video_index.get_unique(f"{prefer_split}/{sample_id}", prefer_prefix=prefer_split)
            if vpath is None:
                vpath = video_index.get_unique(f"{prefer_split}/{asset_id}", prefer_prefix=prefer_split)
        if vpath is None:
            n_missing_video += 1
            continue

        dpath = depth_index.get_unique(sample_id, prefer_prefix=prefer_split)
        if dpath is None:
            dpath = depth_index.get_unique(asset_id, prefer_prefix=prefer_split)
        if dpath is None and prefer_split:
            dpath = depth_index.get_unique(f"{prefer_split}/{sample_id}", prefer_prefix=prefer_split)
            if dpath is None:
                dpath = depth_index.get_unique(f"{prefer_split}/{asset_id}", prefer_prefix=prefer_split)
        if dpath is None:
            n_missing_depth += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # read clip (video file o frames dir)
        is_frames_dir = vpath.is_dir()
        if is_frames_dir:
            fps = float(r.get("_fps", float("nan")))
            if not np.isfinite(fps) or fps <= 0:
                raise RuntimeError(
                    f"[FPS] Missing/invalid fps for {sample_id}. "
                    f"Revisa --meta-csv o habilita fallback explícitamente."
                )
            frame_paths = _list_frame_paths(vpath)
            if not frame_paths:
                n_missing_video += 1
                continue
            T = int(len(frame_paths))
            video_np = None
            frame_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        else:
            try:
                video_t, _, info = read_video(str(vpath), pts_unit="sec")
                fps = float(info.get("video_fps", 30.0))
                video_np = video_t.numpy()
            except Exception:
                n_missing_video += 1
                continue

        if not is_frames_dir:
            if video_np.size == 0 or video_np.shape[0] == 0:
                n_missing_video += 1
                continue
            T = int(video_np.shape[0])
        duration = float(T) / max(float(fps), 1e-6)
        durations.append(duration)

        # DoTA __pre: el "negativo" es el tramo pre-accidente del mismo clip.
        # Recortamos la duración a t_event(base_id) para no incluir evidencia post-accidente.
        duration_use = duration
        forced_duration = r.get("_force_duration_s", None)
        if forced_duration is not None:
            try:
                forced_duration = float(forced_duration)
            except Exception:
                forced_duration = None
        if forced_duration is not None and np.isfinite(forced_duration) and forced_duration > 0:
            duration_use = min(duration_use, float(forced_duration))
        if asset_suffix == "pre":
            tev_base = base_event_time.get(asset_id, None)
            if tev_base is not None and np.isfinite(float(tev_base)):
                duration_use = min(duration_use, max(0.0, float(tev_base) - TIME_EPS))

        # Sanity: time_of_event en segundos
        if target == 1:
            if t_event is None:
                n_pos_invalid_event += 1
            else:
                tev = float(t_event)
                if not (0.0 <= tev <= float(duration_use) + 1e-3):
                    n_pos_invalid_event += 1
                    t_event = None

        centers = _compute_centers(duration=duration_use, hist_s=hist_s, stride_s=stride_s)
        if not centers:
            n_missing_video += 1
            continue

        a_start = r.get("_anom_start_s", None)
        a_end = r.get("_anom_end_s", None)
        try:
            a_start_f = float(a_start) if a_start is not None else float("nan")
            a_end_f = float(a_end) if a_end is not None else float("nan")
        except Exception:
            a_start_f, a_end_f = float("nan"), float("nan")
        has_anom_interval = np.isfinite(a_start_f) and np.isfinite(a_end_f) and (a_end_f >= a_start_f)

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

            if is_frames_dir:
                rgb_clip = np.stack(
                    [_cache_get_frame(frame_paths, int(j), frame_cache) for j in idx.tolist()],
                    axis=0,
                )
            else:
                rgb_clip = video_np[idx]
            rgb_clips.append(rgb_clip)

            if dpath is not None:
                d_clip = load_depth_clip_nearest(dpath, idx, frames_per_clip, rgb_fps=float(fps))
            else:
                d_clip = np.zeros((frames_per_clip, 224, 224, 1), dtype=np.float32)
            depth_clips.append(d_clip)

            # Tiempo real de decisión coherente con los frames vistos (hasta end_idx inclusive).
            # Usamos el "end_time" como tiempo tras observar el frame end_idx.
            end_time = float(end_idx + 1) / max(float(fps), 1e-6)
            window_times.append(end_time)

        clip_probs: List[float] = []
        with torch.no_grad():
            for i in range(0, len(rgb_clips), batch_size):
                batch_rgb = rgb_clips[i:i + batch_size]
                batch_depth = depth_clips[i:i + batch_size]

                inputs = processor(batch_rgb, return_tensors="pt")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                d_np = np.stack(batch_depth, axis=0)
                d_t = torch.from_numpy(d_np).float()
                d_t = d_t[..., 0]  # [B,T,H,W]

                d_t = torch.clamp(d_t, 0.0, 150.0)
                d_t = torch.log1p(d_t) / 5.0
                d_t = d_t.unsqueeze(1).contiguous()  # [B,1,T,H,W]
                d_t = d_t.to(device, non_blocking=True)

                logits = model(
                    pixel_values_videos=inputs["pixel_values_videos"],
                    depth_videos=d_t,
                )
                probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                clip_probs.extend(probs)

        # DoTA window-level labels/scores (para AUC/AP) usando el tiempo de decisión de cada ventana.
        if has_anom_interval and clip_probs:
            for t_w, p_w in zip(window_times, clip_probs):
                # Etiquetado ventana-level coherente con "tiempo de decisión":
                # end_time = (end_idx+1)/fps => en end_time==anomaly_start_s aún NO hemos visto el primer frame anómalo.
                tw = float(t_w)
                lab = 1 if ((tw > (a_start_f + TIME_EPS)) and (tw <= (a_end_f + TIME_EPS))) else 0
                window_labels.append(int(lab))
                window_scores.append(float(p_w))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        per_video_times.append(t1 - t0)

        cp = np.asarray(clip_probs, dtype=np.float32)
        wt = np.asarray(window_times, dtype=np.float32)

        # pre-event score (anticipation)
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

        # anytime score (clasificación por vídeo)
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

        # protocolo: metrics de anticipation -> (negativos) + (positivos con t_event válido)
        if (target == 0) or (t_event is not None):
            all_scores.append(float(pre_score))
            all_targets.append(target)
            all_ids.append(sample_id)

        pred_rows.append({"id": sample_id, "target": target, "score": f"{float(pre_score):.6f}"})

        # TTA
        if (target == 1) and (t_event is not None):
            n_pos_with_event += 1
            tta, hit = compute_tta_for_video(window_times, clip_probs, float(t_event), tta_thr)
            if hit:
                n_pos_hits_before += 1
                tta_hit_values.append(float(tta))
            tta_values.append(float(tta))

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

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    base_ap = float(n_pos) / float(max(1, n_pos + n_neg))

    ap = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    y_true_any = np.asarray(all_targets_anytime, dtype=np.int64)
    y_score_any = np.asarray(all_scores_anytime, dtype=np.float32)
    ap_any = average_precision_score(y_true_any, y_score_any) if len(np.unique(y_true_any)) > 1 else float("nan")
    auc_any = roc_auc_score(y_true_any, y_score_any) if len(np.unique(y_true_any)) > 1 else float("nan")

    win_ap = float("nan")
    win_auc = float("nan")
    win_base_ap = float("nan")
    if window_labels and len(set(window_labels)) > 1:
        y_w = np.asarray(window_labels, dtype=np.int64)
        s_w = np.asarray(window_scores, dtype=np.float32)
        win_ap = float(average_precision_score(y_w, s_w))
        win_auc = float(roc_auc_score(y_w, s_w))
        win_base_ap = float(y_w.mean())

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
        mtta_hits_only = float(np.mean(tta_hit_values)) if tta_hit_values else float("nan")
        recall_before = n_pos_hits_before / max(1, n_pos_with_event)
    else:
        mtta = float("nan")
        mtta_hits_only = float("nan")
        recall_before = 0.0

    if per_video_times:
        avg_t = float(np.mean(per_video_times))
        avg_dur = float(np.mean(durations)) if durations else float("nan")
        xrt = avg_dur / avg_t if (avg_t > 0 and not np.isnan(avg_dur)) else float("nan")
    else:
        avg_t = float("nan")
        xrt = float("nan")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "target", "score"])
            w.writeheader()
            w.writerows(pred_rows)
        print(f"[CSV] Predicciones guardadas en: {out_csv}")

    # Umbral @R*: sólo es interpretable cuando hay pos y neg. Si no, exigimos override.
    if thr_r80_override is None:
        thr_r80 = float("nan") if (n_pos == 0 or n_neg == 0) else find_threshold_for_recall(y_true, y_score, target_recall=target_recall)
    else:
        thr_r80 = float(thr_r80_override)

    if np.isfinite(thr_r80):
        y_pred_r80 = (y_score >= thr_r80).astype(np.int64)
        acc_r80 = accuracy_score(y_true, y_pred_r80)
        prec_r80 = precision_score(y_true, y_pred_r80, zero_division=0)
        rec_r80 = recall_score(y_true, y_pred_r80, zero_division=0)
        f1_r80 = f1_score(y_true, y_pred_r80, zero_division=0)
    else:
        acc_r80 = float("nan")
        prec_r80 = float("nan")
        rec_r80 = float("nan")
        f1_r80 = float("nan")

    if pos_window_times_list and np.isfinite(thr_r80):
        tta_vals_r80: List[float] = []
        tta_hit_vals_r80: List[float] = []
        hits_before_r80 = 0
        for w_times, c_probs, tev in zip(pos_window_times_list, pos_clip_probs_list, pos_time_of_event_list):
            tta_r, hit_r = compute_tta_for_video(w_times, c_probs, tev, thr_r80)
            if hit_r:
                hits_before_r80 += 1
                tta_hit_vals_r80.append(float(tta_r))
            tta_vals_r80.append(float(tta_r))

        mtta_r80 = float(np.mean(tta_vals_r80)) if tta_vals_r80 else float("nan")
        mtta_r80_hits_only = float(np.mean(tta_hit_vals_r80)) if tta_hit_vals_r80 else float("nan")
        rec_before_r80 = hits_before_r80 / max(1, len(tta_vals_r80))
    else:
        mtta_r80 = float("nan")
        mtta_r80_hits_only = float("nan")
        rec_before_r80 = 0.0

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
                tta_thr_i, hit_thr_i = compute_tta_for_video(w_times, c_probs, tev, float(thr))
                hits += int(hit_thr_i)
                tta_vals.append(float(tta_thr_i))

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
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_ap": base_ap,

        "ap": ap,
        "auc": auc,
        "ap_anytime": ap_any,
        "auc_anytime": auc_any,
        "win_ap": win_ap,
        "win_auc": win_auc,
        "win_base_ap": win_base_ap,
        "n_windows": int(len(window_scores)),
        "n_windows_pos": int(sum(window_labels)),
        "n_windows_neg": int(len(window_labels) - sum(window_labels)),

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

def _load_processor_and_cfg(hf_repo: str):
    try:
        processor = AutoVideoProcessor.from_pretrained(hf_repo, local_files_only=True)
        cfg = AutoConfig.from_pretrained(hf_repo, local_files_only=True)
        return processor, cfg
    except Exception as e:
        print(f"[HF] local_files_only falló ({type(e).__name__}). Reintentando online...")
        processor = AutoVideoProcessor.from_pretrained(hf_repo)
        cfg = AutoConfig.from_pretrained(hf_repo)
        return processor, cfg


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--video-root", type=str, required=True,
                    help="Root de vídeos o root de frames (DoTA_seg/frames).")
    ap.add_argument("--frames-root", type=str, default=None,
                    help="Opcional: root explícito de frames (override de --video-root).")
    ap.add_argument("--depth-root", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="preds_dota_tfdepth_robust_sliding.csv")
    ap.add_argument("--meta-csv", type=str, default="/home/ander/V-JEPA 2/data/metadata/DoTA/ccd_meta.csv",
                    help="Path a ccd_meta.csv para fps/duración (modo frames).")
    ap.add_argument(
        "--annotations-dir",
        type=str,
        default="/home/ander/V-JEPA 2/data/metadata/DoTA/DoTA_annotations/annotations",
        help="Directorio con JSONs de anotación por clip (<id>.json) para filtros ego/ignore.",
    )
    ap.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Filtra por canal del JSON (p.ej. CarCrashesTime, AnAn). Requiere --annotations-dir.",
    )
    ap.add_argument(
        "--ego-only",
        action="store_true",
        help="Filtra a clips ego-involve=True e ignore=False usando --annotations-dir (aplica a base_id para __pre).",
    )
    ap.add_argument(
        "--add-pre-negatives",
        action="store_true",
        help="(DoTA) Añade negativos sintéticos tipo '<id>__pre' para cada positivo (necesario si tu CSV no trae negativos).",
    )
    ap.add_argument(
        "--add-synth-negatives",
        action="store_true",
        help=(
            "(DoTA) Añade negativos sintéticos 'early clip': recorta a los primeros --synth-neg-seconds segundos "
            "para los positivos cuyo tiempo (alert/evento) sea >= --synth-neg-min-time. Inspirado en BADAS paper."
        ),
    )
    ap.add_argument("--synth-neg-seconds", type=float, default=4.0)
    ap.add_argument("--synth-neg-min-time", type=float, default=4.5)
    ap.add_argument(
        "--synth-neg-max",
        type=int,
        default=None,
        help="Si se especifica, limita el número de negativos sintéticos añadidos (útil para igualar n de papers).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed para muestreo reproducible de negativos sintéticos.")
    ap.add_argument(
        "--synth-neg-time-source",
        type=str,
        default="time_of_alert",
        choices=["time_of_alert", "time_of_event"],
        help="Qué timestamp usar para decidir si un clip admite negativo early.",
    )
    ap.add_argument(
        "--allow-fps-fallback",
        action="store_true",
        help="(modo frames) Permite usar fps=10.0 cuando falta en --meta-csv (NO recomendado).",
    )

    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--stride-s", type=float, default=0.5)

    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean", "topk"])
    ap.add_argument("--top-k", type=int, default=5)

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tta-thr", type=float, default=0.5)
    ap.add_argument("--target-recall", type=float, default=0.8)
    ap.add_argument(
        "--thr-r80",
        type=float,
        default=None,
        help=(
            "Si lo pasas, usa este umbral fijo para las métricas @R (Acc/F1/TTA@R...). "
            "Recomendado para evaluar DoTA test (que sólo tiene positivos) usando un umbral calibrado en val."
        ),
    )

    ap.add_argument(
        "--prefer-split",
        type=str,
        default="auto",
        choices=["auto", "train", "val", "test", "training", "testing", "none"],
        help="Para resolver colisiones de IDs entre splits cuando --video-root es la raíz.",
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
    random.seed(int(args.seed))

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
    processor, cfg = _load_processor_and_cfg(hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))

    train_mod = import_train_module(train_script)
    ModelCls = pick_model_class(train_mod, forced_class=args.model_class)
    model = instantiate_model_from_ckpt(ModelCls, ckpt_args)

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
    print(f"DoTA : prefer_split={prefer_split}")
    print("=" * 70)

    rows = read_csv_with_event(csv_path)
    print(f"[DATA] Filas CSV: {len(rows)}")

    # ----------------------------------------------------------
    # (Opcional) Filtro ego-centric + síntesis de negativos para DoTA.
    # Nota: DoTA oficial trae sólo positivos en test; papers suelen sintetizar negativos.
    # ----------------------------------------------------------
    annotations_dir = Path(args.annotations_dir)
    if (args.ego_only or args.add_pre_negatives or args.add_synth_negatives) and not annotations_dir.exists():
        raise FileNotFoundError(f"--annotations-dir no existe: {annotations_dir}")

    if args.ego_only:
        import json
        keep: List[Dict[str, Any]] = []
        n_drop = 0
        for rr in rows:
            base_id, _ = split_dota_id(rr["id"])
            ann_path = annotations_dir / f"{base_id}.json"
            if not ann_path.exists():
                n_drop += 1
                continue
            ann = json.load(ann_path.open("r"))
            if bool(ann.get("ignore", False)):
                n_drop += 1
                continue
            if not bool(ann.get("ego_involve", False)):
                n_drop += 1
                continue
            if args.channel is not None and str(ann.get("channel", "")) != str(args.channel):
                n_drop += 1
                continue
            keep.append(rr)
        rows = keep
        print(f"[EGO_ONLY] keep={len(rows)} drop={n_drop}")
    elif args.channel is not None:
        import json
        keep: List[Dict[str, Any]] = []
        n_drop = 0
        for rr in rows:
            base_id, _ = split_dota_id(rr["id"])
            ann_path = annotations_dir / f"{base_id}.json"
            if not ann_path.exists():
                n_drop += 1
                continue
            ann = json.load(ann_path.open("r"))
            if str(ann.get("channel", "")) != str(args.channel):
                n_drop += 1
                continue
            keep.append(rr)
        rows = keep
        print(f"[CHANNEL] channel={args.channel} keep={len(rows)} drop={n_drop}")

    if args.add_pre_negatives:
        extra: List[Dict[str, Any]] = []
        for rr in rows:
            if int(rr["target"]) != 1:
                continue
            base_id, _ = split_dota_id(rr["id"])
            extra.append({"id": f"{base_id}__pre", "target": 0, "time_of_event": None, "time_of_alert": None})
        rows = rows + extra
        print(f"[ADD_PRE_NEG] added={len(extra)} total={len(rows)}")

    if args.add_synth_negatives:
        extra: List[Dict[str, Any]] = []
        used_alert = 0
        for rr in rows:
            if int(rr["target"]) != 1:
                continue
            base_id, _ = split_dota_id(rr["id"])
            if args.synth_neg_time_source == "time_of_alert":
                t_ref = rr.get("time_of_alert", None)
                if t_ref is not None:
                    used_alert += 1
                if t_ref is None:
                    t_ref = rr.get("time_of_event", None)
            else:
                t_ref = rr.get("time_of_event", None)
            if t_ref is None:
                continue
            if float(t_ref) >= float(args.synth_neg_min_time):
                extra.append(
                    {
                        "id": f"{base_id}__neg{int(round(args.synth_neg_seconds))}s",
                        "target": 0,
                        "time_of_event": None,
                        "time_of_alert": None,
                        "_force_duration_s": float(args.synth_neg_seconds),
                    }
                )
        if args.synth_neg_max is not None and args.synth_neg_max > 0 and len(extra) > int(args.synth_neg_max):
            rnd = random.Random(int(args.seed))
            extra_sorted = sorted(extra, key=lambda d: str(d.get("id", "")))
            rnd.shuffle(extra_sorted)
            extra = extra_sorted[: int(args.synth_neg_max)]
        rows = rows + extra
        print(
            f"[ADD_SYNTH_NEG] added={len(extra)} total={len(rows)} "
            f"used_time_of_alert={used_alert} seed={args.seed} max={args.synth_neg_max}"
        )

    video_root = Path(args.frames_root) if args.frames_root else Path(args.video_root)
    depth_root = Path(args.depth_root)
    # Auto (rápido): si parece estructura DoTA_seg/frames/<id>/images => modo frames.
    # Esto evita un rglob enorme sobre millones de imágenes.
    using_frames = bool(args.frames_root)
    if not using_frames:
        try:
            for child in video_root.iterdir():
                if child.is_dir() and (child / "images").is_dir():
                    using_frames = True
                    break
        except Exception:
            using_frames = False

    if using_frames:
        video_index = index_frames_dota(video_root)
    else:
        video_index = index_videos_dota(video_root)

    if video_index.n_keys() == 0:
        raise RuntimeError(f"No he encontrado vídeos ni clips de frames bajo: {video_root}")

    depth_index = index_depth_dota(depth_root)

    if depth_index.n_keys() == 0:
        print("[WARN] No se han encontrado .npz en depth-root. Evaluaré con depth=0 (peor rendimiento).")
        print("[HINT] Para precomputar depth en DoTA:")
        if using_frames:
            print(
                f"       python {Path(__file__).parent / 'preprocess_depth_da3_frames.py'} "
                f"--frames-root \"{video_root}\" --out-root \"{depth_root}\" "
                f"--meta-csv \"{args.meta_csv}\""
            )
        else:
            print(f"       python {Path(__file__).parent / 'preprocess_depth_da3.py'} --video-root \"{video_root}\" --out-root \"{depth_root}\"")

    # En modo frames, añadimos fps por sample usando ccd_meta.csv si existe
    if using_frames:
        meta = load_ccd_meta_map(Path(args.meta_csv) if args.meta_csv else None)
        for rr in rows:
            vid = rr["id"]
            asset_id, _ = split_dota_id(vid)
            fps = meta.get(asset_id, {}).get("fps", float("nan"))
            rr["_anom_start_s"] = meta.get(asset_id, {}).get("anomaly_start_s", float("nan"))
            rr["_anom_end_s"] = meta.get(asset_id, {}).get("anomaly_end_s", float("nan"))
            if fps and not np.isnan(fps) and fps > 0:
                rr["_fps"] = float(fps)
            else:
                if args.allow_fps_fallback:
                    rr["_fps"] = 10.0
                else:
                    raise RuntimeError(
                        f"[FPS] Missing/invalid fps for {vid} (asset_id={asset_id}) en {args.meta_csv}. "
                        f"Usa --allow-fps-fallback si quieres forzar fps=10.0."
                    )

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
        thr_r80_override=args.thr_r80,
        out_csv=Path(args.out_csv),
    )

    print("\n========== RESULTADOS DoTA (SLIDING + TTA) ==========")
    print(f"Videos evaluados      : {res.get('n_eval', 0)}")
    print(f"Videos evaluados(any) : {res.get('n_eval_anytime', 0)}")
    print(f"Balance (pos/neg)     : {res.get('n_pos', 0)}/{res.get('n_neg', 0)}")
    if int(res.get("n_neg", 0)) == 0 or int(res.get("n_pos", 0)) == 0:
        print("[WARN] Split sin ambas clases (pos/neg): AP/AUC y métricas tipo Acc/F1 no son interpretables; "
              "usa val (con negativos __pre) para calibrar umbrales y reporta TTA en test.")
    print(f"Videos missing(video) : {res.get('n_missing_video', 0)}")
    print(f"Videos missing(depth) : {res.get('n_missing_depth', 0)}")
    print(f"Pos sin t_event válido: {res.get('n_pos_invalid_event', 0)}")
    print(f"AP_pre_event          : {res.get('ap', float('nan')):.6f}")
    print(f"AUC_pre_event         : {res.get('auc', float('nan')):.6f}")
    print(f"AP_anytime(full)      : {res.get('ap_anytime', float('nan')):.6f}")
    print(f"AUC_anytime(full)     : {res.get('auc_anytime', float('nan')):.6f}")
    print(f"AP_baseline(video)    : {res.get('base_ap', float('nan')):.6f}")
    print(f"AP_window(DoTA)       : {res.get('win_ap', float('nan')):.6f} (base={res.get('win_base_ap', float('nan')):.6f} n_win={res.get('n_windows', 0)} pos={res.get('n_windows_pos', 0)} neg={res.get('n_windows_neg', 0)})")
    print(f"AUC_window(DoTA)      : {res.get('win_auc', float('nan')):.6f}")
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
    if np.isfinite(res.get("thr_r80", float("nan"))):
        print(f"Acc@R{args.target_recall:.2f}             : {res.get('acc_r80', float('nan'))*100:.2f}%")
        print(f"F1@R{args.target_recall:.2f}              : {res.get('f1_r80', float('nan')):.3f} "
              f"(P:{res.get('prec_r80', float('nan')):.2f} R:{res.get('rec_r80', float('nan')):.2f})")
        print(f"TTA@R{args.target_recall:.2f}             : {res.get('mtta_r80', float('nan')):.3f} s")
        print(f"TTA_hits@R{args.target_recall:.2f}        : {res.get('mtta_r80_hits_only', float('nan')):.3f} s")
        print(f"Rec_before_event@R{args.target_recall:.2f}: {res.get('recall_before_event_r80', 0.0)*100:.2f}%")
    else:
        print(f"[WARN] thr_R{args.target_recall:.2f} no definido (split sin ambas clases y sin --thr-r80).")
    print(f"mTTA_curve             : {res.get('mtta_curve', float('nan')):.3f} s "
          f"(max_recall={res.get('max_recall_curve', 0.0):.3f})")
    print("====================================================")


if __name__ == "__main__":
    main()
