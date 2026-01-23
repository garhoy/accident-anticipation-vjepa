#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_nexar_mid_fusion_sliding_tta.py

Eval sliding-window + TTA para CHECKPOINTS entrenados con:
  examples/Depth_VJEPA/mid_fusion.py   (MID-FUSION FiLM)

NO reimplementa el modelo aquí:
  - Importa dinámicamente mid_fusion.py
  - Instancia la clase modelo del train
  - Carga state_dict de tu checkpoint

Métricas:
  - AP_anytime, AUC_anytime
  - Acc/Prec/Rec/F1 @ 0.5
  - thr_R80 (umbral mínimo para recall>=target_recall)
  - Acc/Prec/Rec/F1 @ R80
  - mTTA@tta_thr y Recall_before_event@tta_thr
  - mTTA@R80 y Recall_before_event@R80
  - avg_time_per_video, xRT

Uso:
CUDA_VISIBLE_DEVICES=0 python examples/Depth_VJEPA/eval_nexar_mid_fusion_sliding_tta.py \
  --checkpoint "checkpoints_vjepa2_midfilm_L-4_-3_-2_-1/best_midfilm_-4_-3_-2_-1.pt" \
  --csv "/home/ander/V-JEPA 2/data/metadata/Nexar/extraction_csvs/val.csv" \
  --video-root "/home/ander/V-JEPA 2/data/raw/Nexar/train" \
  --depth-root "/home/ander/BADAS-Open/data/processed/Nexar_DA3_Tensors" \
  --hist-s 5.0 --stride-s 0.5 --agg max --batch-size 16 \
  --tta-thr 0.5 --target-recall 0.8
"""

import os
import sys
import csv
import time
import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
# 0) PATH: asegurar repo root en sys.path (para imports del mid_fusion.py)
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
# 2) INDEX VIDEO / DEPTH
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
    Carga depth .npz y alinea a los frame_indices vía nearest neighbor.
    Devuelve [T, H, W, 1] float32. Fallback zeros.
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

    mask = (probs >= thr) & (times <= time_of_event)
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
# 4) IMPORT DINÁMICO DEL MODELO MID-FUSION
# ==========================================================

def import_train_module(train_script: Path):
    spec = importlib.util.spec_from_file_location("train_midfusion_mod", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No puedo importar: {train_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_model_class(mod, forced_class: Optional[str] = None):
    """
    Elige la clase del modelo principal del train:
      - si forced_class: usa esa
      - si no: busca nn.Module con forward(pixel_values_videos, depth_videos, ...)
        y nombre que contenga VJEPA + (FiLM/Mid/mid)
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

        # descartes obvios
        if name in {"DepthPatchTokenizer", "DenseRGBDepthFusion", "SpatialPooler"}:
            continue

        # forward signature
        try:
            sig = inspect.signature(obj.forward)
            params = list(sig.parameters.keys())
        except Exception:
            continue

        # queremos algo que huela a tu modelo final
        has_rgb = any("pixel_values" in p for p in params)
        has_depth = any("depth" in p for p in params)

        if not (has_rgb and has_depth):
            continue

        score = 0
        low = name.lower()
        if "vjepa" in low: score += 3
        if "film" in low: score += 3
        if "mid" in low: score += 2
        if "binary" in low: score += 1
        candidates.append((score, name, obj))

    if not candidates:
        raise RuntimeError(
            "No he encontrado clase modelo mid-fusion en el train script.\n"
            "Pásame --model-class <NombreClase> o revisa que el modelo tenga forward(pixel_values_videos, depth_videos, ...)."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    print(f"[MODEL_PICK] Using class: {best[1]} (score={best[0]})")
    return best[2]

def instantiate_midfusion_from_ckpt(mod, ModelCls, ckpt_args: Dict[str, Any]):
    """
    Instancia el modelo con args del checkpoint, pero:
      - Filtra kwargs por la firma real del __init__
      - Mapea nombres típicos (film_per_layer -> per_layer, etc.)
    """
    import inspect

    def _accepted_params(cls):
        sig = inspect.signature(cls.__init__)
        return set(sig.parameters.keys())  # incluye "self"

    def _pick_alt(accepted, candidates):
        for c in candidates:
            if c in accepted:
                return c
        return None

    accepted = _accepted_params(ModelCls)
    accepted.discard("self")

    # ---- Canonical args desde checkpoint (los tuyos) ----
    hf_repo = ckpt_args.get("hf_repo", "facebook/vjepa2-vitl-fpc16-256-ssv2")
    depth_dim = int(ckpt_args.get("depth_dim", 128))
    spatial_heads = int(ckpt_args.get("spatial_heads", 1))
    unfreeze_blocks = int(ckpt_args.get("unfreeze_blocks", 0))
    encoder_ckpt = ckpt_args.get("encoder_ckpt", None)

    inject_layers = ckpt_args.get("inject_layers", "-4,-3,-2,-1")
    if isinstance(inject_layers, (list, tuple)):
        inject_layers = ",".join(str(x) for x in inject_layers)

    film_hidden = int(ckpt_args.get("film_hidden", 256))
    film_scale = float(ckpt_args.get("film_scale", 0.05))
    film_per_layer = bool(ckpt_args.get("film_per_layer", False))

    # ---- Construimos kwargs “canónicos” ----
    kwargs = {
        "hf_repo": hf_repo,
        "depth_dim": depth_dim,
        "spatial_heads": spatial_heads,
        "unfreeze_blocks": unfreeze_blocks,
        "encoder_ckpt": encoder_ckpt,

        "inject_layers": inject_layers,
        "film_hidden": film_hidden,
        "film_scale": film_scale,
        "film_per_layer": film_per_layer,
    }

    # ---- Mapeo de sinónimos según firma real ----
    # 1) film_per_layer suele llamarse "per_layer" o similar
    if "film_per_layer" not in accepted:
        alt = _pick_alt(accepted, ["per_layer", "film_perlayer", "per_layer_film", "film_per_layer_flag"])
        if alt is not None:
            kwargs[alt] = kwargs.pop("film_per_layer")
        else:
            kwargs.pop("film_per_layer", None)

    # 2) film_hidden a veces se llama distinto
    if "film_hidden" not in accepted:
        alt = _pick_alt(accepted, ["film_mlp_hidden", "film_dim", "hidden", "adapter_hidden"])
        if alt is not None:
            kwargs[alt] = kwargs.pop("film_hidden")
        else:
            kwargs.pop("film_hidden", None)

    # 3) film_scale a veces se llama distinto
    if "film_scale" not in accepted:
        alt = _pick_alt(accepted, ["scale", "gamma_scale", "film_gamma_scale"])
        if alt is not None:
            kwargs[alt] = kwargs.pop("film_scale")
        else:
            kwargs.pop("film_scale", None)

    # 4) inject_layers puede llamarse "inject_at" o similar
    if "inject_layers" not in accepted:
        alt = _pick_alt(accepted, ["inject_at", "layers_to_inject", "inject_indices"])
        if alt is not None:
            kwargs[alt] = kwargs.pop("inject_layers")
        else:
            kwargs.pop("inject_layers", None)

    # ---- Filtrado final: sólo lo que acepta el __init__ ----
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    print("[INIT_KWARGS] Using kwargs:", kwargs)

    try:
        return ModelCls(**kwargs)
    except TypeError as e:
        raise RuntimeError(
            f"No puedo instanciar {ModelCls.__name__} con kwargs filtrados.\n"
            f"Accepted params: {sorted(list(accepted))}\n"
            f"Kwargs usados: {kwargs}\n"
            f"Error: {e}"
        )


# ==========================================================
# 5) EVAL SLIDING
# ==========================================================

def evaluate_split(
    rows: List[Dict[str, Any]],
    model: nn.Module,
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
    target_recall: float,
    out_csv: Optional[Path] = None,
) -> Dict[str, Any]:

    iterator = tqdm(rows, desc="Eval", unit="vid") if tqdm else rows

    all_scores: List[float] = []
    all_targets: List[int] = []
    all_ids: List[str] = []
    pred_rows: List[Dict[str, Any]] = []

    n_missing = 0

    # TTA @ tta_thr
    tta_values: List[float] = []
    n_pos_with_event = 0
    n_pos_hits_before = 0

    # Para TTA@R80
    pos_window_times_list: List[List[float]] = []
    pos_clip_probs_list: List[List[float]] = []
    pos_time_of_event_list: List[float] = []

    per_video_times: List[float] = []
    durations: List[float] = []

    model.eval()

    for r in iterator:
        vid = r["id"]
        target = int(r["target"])
        t_event = r.get("time_of_event", None)

        vpath = video_map.get(vid, None)
        if vpath is None:
            n_missing += 1
            continue
        dpath = depth_map.get(vid, None)

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

        T = video.shape[0]
        duration = T / max(fps, 1e-6)
        durations.append(duration)

        # centers
        if duration <= hist_s:
            centers = [0.5 * duration]
        else:
            n_steps = int(np.floor((duration - hist_s) / stride_s)) + 1
            centers = [hist_s + i * stride_s for i in range(max(1, n_steps))]
            if centers and centers[-1] < duration - 0.25:
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

            window_times.append(end_s)

        if not rgb_clips:
            n_missing += 1
            continue

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

                # Normalización EXACTA a train
                d_t = torch.clamp(d_t, 0.0, 150.0)
                d_t = torch.log1p(d_t) / 5.0
                d_t = d_t.unsqueeze(1).contiguous()    # [B,1,T,H,W]
                d_t = d_t.to(device, non_blocking=True)

                logits = model(
                    pixel_values_videos=inputs["pixel_values_videos"],
                    depth_videos=d_t,
                )  # [B] logits (ya clip-level)
                probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                clip_probs.extend(probs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        per_video_times.append(t1 - t0)

        # agregación video
        cp = np.asarray(clip_probs, dtype=np.float32)
        if cp.size == 0:
            video_score = 0.0
        else:
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

        pred_rows.append({"id": vid, "target": target, "score": f"{video_score:.6f}"})

        # TTA
        if (target == 1) and (t_event is not None):
            n_pos_with_event += 1
            tta, hit = compute_tta_for_video(window_times, clip_probs, float(t_event), tta_thr)
            if hit:
                n_pos_hits_before += 1
            tta_values.append(tta)

            pos_window_times_list.append(window_times)
            pos_clip_probs_list.append(clip_probs)
            pos_time_of_event_list.append(float(t_event))

    if not all_scores:
        return {"n_eval": 0, "n_missing": n_missing}

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_score = np.asarray(all_scores, dtype=np.float32)

    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

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
        recall_before = n_pos_hits_before / max(1, n_pos_with_event)
    else:
        mtta = float("nan")
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
# 6) MAIN
# ==========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--video-root", type=str, required=True)
    ap.add_argument("--depth-root", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="preds_nexar_midfusion_sliding.csv")

    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--stride-s", type=float, default=0.5)

    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean", "topk"])
    ap.add_argument("--top-k", type=int, default=5)

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tta-thr", type=float, default=0.5)

    ap.add_argument("--target-recall", type=float, default=0.8)

    # dónde está el train script mid_fusion.py
    ap.add_argument("--train-script", type=str, default=str(Path(__file__).parent / "mid_fusion.py"))
    ap.add_argument("--model-class", type=str, default=None,
                    help="Si el auto-pick falla, pon aquí el nombre exacto de la clase del modelo en mid_fusion.py")

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

    # repo y cfg
    hf_repo = ckpt_args.get("hf_repo", "facebook/vjepa2-vitl-fpc16-256-ssv2")
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    cfg = AutoConfig.from_pretrained(hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))

    # construir modelo desde mid_fusion.py
    mid_mod = import_train_module(train_script)
    ModelCls = pick_model_class(mid_mod, forced_class=args.model_class)
    model = instantiate_midfusion_from_ckpt(mid_mod, ModelCls, ckpt_args)

    # load state
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    # HARD CHECK: no aceptar mismatches “críticos”
    critical_substr = ("film", "adapter", "modulat", "temporal_head", "depth", "vjepa2")
    bad_missing = [k for k in missing if any(s in k.lower() for s in critical_substr)]
    bad_unexpected = [k for k in unexpected if any(s in k.lower() for s in critical_substr)]

    if bad_missing or bad_unexpected:
        print("[ERR] mismatch crítico al cargar checkpoint.")
        print("missing (first 30):", bad_missing[:30])
        print("unexpected (first 30):", bad_unexpected[:30])
        raise RuntimeError("No voy a evaluar un modelo distinto al entrenado. Arregla clase/args y vuelve a intentar.")

    model.to(device)
    model.eval()

    # stats
    model_size_mb = get_model_size_mb(model)
    n_params = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 70)
    print(f"MODEL: {ModelCls.__name__} | Params: {n_params/1e6:.1f}M | Size: {model_size_mb:.1f} MB")
    print(f"CFG  : frames_per_clip={frames_per_clip} | hist={args.hist_s}s | stride={args.stride_s}s")
    print(f"EVAL : batch={args.batch_size} | agg={args.agg}-{args.top_k} | tta_thr={args.tta_thr} | R@{args.target_recall}")
    print("=" * 70)

    rows = read_csv_with_event(csv_path)
    print(f"[DATA] Vídeos en CSV: {len(rows)}")

    video_map = index_videos(Path(args.video_root))
    depth_map = index_depth(Path(args.depth_root))

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
        target_recall=args.target_recall,
        out_csv=Path(args.out_csv),
    )

    print("\n========== RESULTADOS MID-FUSION (SLIDING + TTA) ==========")
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
    print("----------------------------------------------------------")
    print(f"thr_R{args.target_recall:.2f}          : {res.get('thr_r80', float('nan')):.3f}")
    print(f"Acc@R{args.target_recall:.2f}          : {res.get('acc_r80', 0.0)*100:.2f}%")
    print(f"F1@R{args.target_recall:.2f}           : {res.get('f1_r80', 0.0):.3f} "
          f"(P:{res.get('prec_r80', 0.0):.2f} R:{res.get('rec_r80', 0.0):.2f})")
    print(f"mTTA@R{args.target_recall:.2f}         : {res.get('mtta_r80', float('nan')):.3f} s")
    print(f"Rec_before_event@R{args.target_recall:.2f}: {res.get('recall_before_event_r80', 0.0)*100:.2f}%")
    print("==========================================================")

if __name__ == "__main__":
    main()
