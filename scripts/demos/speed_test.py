#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speed_test_anytime_fair.py

Benchmark "paper-grade" para Setting B (anytime / sliding window) con métricas comparables.

Pipelines soportados:
  - RGB        : VJEPA2TemporalBinary (sliding windows)
  - RGBD*      : VJEPA2DepthTransformerBinary + depth precomputada (.npz) (sliding)
  - RGBD online: DepthAnything3 (depth por frames únicos) + VJEPA2DepthTransformerBinary (sliding) [opcional]
  - BADAS-Open : baseline caja negra (predict(path)) [opcional]

Métricas (por pipeline):
  - E2E ms/video (end-to-end real)
  - Breakdown: decode / window-build / preproc / depth / forward_sys
  - GPU-only ms/pred (CUDA events, solo forward)
  - ms/pred (system) y pred/s (system)
  - xRealTime = video_duration / proc_time   ( >1 => faster-than-real-time )
  - native_fps_throughput = native_frames / proc_time
  - effective_fps_processed = (n_windows * frames_per_clip) / proc_time  (tu "budget" real)
  - VRAM pico (MiB)

Notas de fairness:
  1) "FPS" aquí NO es videos/s disfrazado. Reportamos pred/s, xRT y throughput de frames.
  2) ms/frame para BADAS es engañoso si hay solapamiento interno. Reportamos xRT y, si se puede, ms/pred.
  3) DA3 online se hace "per-unique-frame" (correcto). "per-window" es trampa (penaliza depth por redundancia).

Uso típico:
  CUDA_VISIBLE_DEVICES=0 python speed_test_anytime_fair.py \
    --video-root "/home/ander/V-JEPA 2/data/raw/Nexar/val" \
    --hf-repo "facebook/vjepa2-vitl-fpc16-256-ssv2" \
    --rgb-ckpt  "/home/ander/.../best_rgb.pt" \
    --rgbd-ckpt "/home/ander/.../best_rgbd.pt" \
    --depth-root-pre "/home/ander/BADAS-Open/data/processed/Nexar_DA3_Tensors/val" \
    --hist-s 5.0 --stride-s 0.5 --batch-size 16 --max-videos 30 \
    --amp \
    --include-badas

  (Opcional DA3 online)
    --include-rgbd-online --depth-model-name "depth-anything/DA3METRIC-LARGE"
"""

import os
import sys
import time
import json
import math
import argparse
import gc
import inspect
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import cv2
from tabulate import tabulate
from torchvision.io import read_video
from transformers import AutoVideoProcessor, AutoConfig

# ---------------------------------------------------------------------
# 0) CUDNN / AMP
# ---------------------------------------------------------------------
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}

# ---------------------------------------------------------------------
# 1) SETUP PATHS (blindado al repo BADAS-Open)
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()


def _find_repo_root(start: Path) -> Path:
    """
    Encuentra la raíz del repo (buscando pyproject.toml / requirements.txt / .git).
    """
    for p in (start,) + tuple(start.parents):
        if (p / "pyproject.toml").exists() or (p / "requirements.txt").exists() or (p / ".git").exists():
            return p
    # Fallback razonable para scripts/demos/speed_test.py -> repo root = parents[2]
    return start.parents[2] if len(start.parents) >= 3 else start.parent


REPO_ROOT = _find_repo_root(CURRENT_FILE)
SCRIPTS_DIR = REPO_ROOT / "scripts"
VJEPA_CORE_DIR = SCRIPTS_DIR / "vjepa_core"
DEPTH_VJEPA_DIR = SCRIPTS_DIR / "depth_vjepa"

for p in (REPO_ROOT, SCRIPTS_DIR, VJEPA_CORE_DIR, DEPTH_VJEPA_DIR):
    if p.exists():
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)

# ---------------------------------------------------------------------
# 2) IMPORT MODELS
# ---------------------------------------------------------------------
try:
    # Preferimos los "train_*" si están disponibles (match exacto con checkpoints).
    # Fallback a los "eval_*" si no (más ligeros y suelen no requerir wandb).
    try:
        from train_nexar_vjepa_ft import VJEPA2TemporalBinary
    except Exception:
        from eval_nexar_vjepa_ft import VJEPA2TemporalBinary

    # RGBD (2 variantes): (a) depth_net per-frame (b) patch/tubelet + DenseFusion
    try:
        from train_nexar_depth_general import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) == "wandb":
            import types
            sys.modules["wandb"] = types.SimpleNamespace(
                init=lambda *a, **k: None,
                log=lambda *a, **k: None,
                finish=lambda *a, **k: None,
            )
            from train_nexar_depth_general import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet
        else:
            from eval_nexar_fusion_depth import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet
    except Exception:
        from eval_nexar_fusion_depth import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet

    try:
        from train_nexar_patchwise import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) == "wandb":
            import types
            sys.modules["wandb"] = types.SimpleNamespace(
                init=lambda *a, **k: None,
                log=lambda *a, **k: None,
                finish=lambda *a, **k: None,
            )
            from train_nexar_patchwise import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise
        else:
            from eval_nexar_fusion_depth_patchwise import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise
    except Exception:
        from eval_nexar_fusion_depth_patchwise import VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise

    # Mid-fusion (opcional)
    try:
        from mid_fusion import VJEPA2DepthMidFiLMBinary  # type: ignore
        HAS_MIDFUSION = True
    except ModuleNotFoundError as e:
        # speed_test no debería necesitar wandb; si falta, metemos un stub mínimo para poder importar el modelo.
        if str(getattr(e, "name", "")) == "wandb":
            import types
            sys.modules["wandb"] = types.SimpleNamespace(
                init=lambda *a, **k: None,
                log=lambda *a, **k: None,
                finish=lambda *a, **k: None,
            )
            try:
                from mid_fusion import VJEPA2DepthMidFiLMBinary  # type: ignore
                HAS_MIDFUSION = True
            except Exception:
                HAS_MIDFUSION = False
        else:
            HAS_MIDFUSION = False
    except Exception:
        HAS_MIDFUSION = False

    print("[INIT] Import modelos OK.")
except Exception as e:
    print(f"[CRITICAL] No puedo importar tus modelos. Error: {e}")
    sys.exit(1)

# DepthAnything3 (opcional)
try:
    from depth_anything_3.api import DepthAnything3
    HAS_DA3 = True
except Exception:
    HAS_DA3 = False

# BADAS (opcional)
try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


# ---------------------------------------------------------------------
# 3) HELPERS
# ---------------------------------------------------------------------
def _normalize_ckpt_list(x: Any) -> List[str]:
    """
    Normaliza argumentos de CLI que pueden venir como str, lista, o repetidos con comas.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out: List[str] = []
        for it in x:
            out.extend(_normalize_ckpt_list(it))
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts
    return [str(x)]


def _short_ckpt_name(path: str) -> str:
    p = Path(path)
    # Nombre compacto pero informativo
    if p.suffix == ".pt":
        return p.stem
    return p.name


def _safe_construct(cls, **kwargs):
    """
    Instancia cls filtrando kwargs por firma (para soportar variantes train/eval).
    """
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filt = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filt)


class GPUTimer:
    """CUDA event timer. Devuelve ms (solo kernels GPU)."""
    def __init__(self, device: torch.device):
        self.use_cuda = (device.type == "cuda")
        if self.use_cuda:
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.s.record()

    def stop_ms(self) -> float:
        if self.use_cuda:
            self.e.record()
            torch.cuda.synchronize()
            return float(self.s.elapsed_time(self.e))
        return 0.0


def list_videos(root: Path, max_videos: int) -> List[Path]:
    vids: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix in VIDEO_EXTS:
            vids.append(p)
    vids = sorted(list({v.resolve() for v in vids}))
    if max_videos > 0:
        vids = vids[:max_videos]
    print(f"[INFO] videos: {len(vids)}")
    return vids


def index_depth_npz(depth_root: Path) -> Dict[str, Path]:
    """
    Indexa recursivamente `*.npz` y devuelve map {stem -> path}.
    Esto soporta out-roots que preservan la estructura de carpetas.
    """
    depth_map: Dict[str, Path] = {}
    for p in depth_root.rglob("*.npz"):
        # Si hay colisiones de stem, nos quedamos con el primero encontrado.
        if p.stem not in depth_map:
            depth_map[p.stem] = p
    return depth_map


def resolve_depth_npz_for_video(
    depth_root: Path,
    video_root: Path,
    video_path: Path,
    depth_index: Optional[Dict[str, Path]] = None,
) -> Optional[Path]:
    """
    Resuelve el .npz correspondiente a un vídeo.
    Preferencia:
      1) out-root que preserva estructura: depth_root / relative(video_path, video_root) con .npz
      2) flat: depth_root / f\"{stem}.npz\"
      3) índice por stem (fallback)
    """
    vp = video_path.resolve()
    vr = video_root.resolve()

    try:
        rel = vp.relative_to(vr)
        cand = (depth_root / rel).with_suffix(".npz")
        if cand.exists():
            return cand
    except Exception:
        pass

    cand = depth_root / f"{video_path.stem}.npz"
    if cand.exists():
        return cand

    if depth_index is not None:
        return depth_index.get(video_path.stem, None)

    return None


def compute_window_end_times(duration_s: float, hist_s: float, stride_s: float) -> List[float]:
    """
    Replica tu lógica de eval: end_s avanza desde hist_s en stride_s, y fuerza cubrir el final.
    """
    if duration_s <= 1e-6:
        return [0.0]

    if duration_s <= hist_s:
        return [0.5 * duration_s]

    n_steps = int(np.floor((duration_s - hist_s) / stride_s)) + 1
    ends = [hist_s + i * stride_s for i in range(max(1, n_steps))]
    if ends[-1] < duration_s - 0.25:
        ends.append(duration_s)
    else:
        ends[-1] = min(ends[-1], duration_s)
    return [min(max(e, 0.0), duration_s) for e in ends]


def frame_indices_for_window(
    T: int,
    fps: float,
    end_s: float,
    hist_s: float,
    frames_per_clip: int,
) -> np.ndarray:
    """
    Devuelve idx (len=frames_per_clip) uniformemente entre start_idx y end_idx.
    """
    duration_s = T / max(fps, 1e-6)
    end_s = min(max(end_s, 0.0), duration_s)
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
    return idx


def load_depth_npz_once(npz_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Devuelve (depth_stack, frame_idx) o None.
      depth_stack: [N,H,W] float16/float32
      frame_idx:   [N] int
    """
    try:
        data = np.load(str(npz_path))
        d = data["depth"]
        fi = data["frame_idx"]
        if d.ndim != 3 or fi.ndim != 1 or d.shape[0] != fi.shape[0] or d.shape[0] == 0:
            return None
        # Asegurar orden por frame_idx
        order = np.argsort(fi)
        fi = fi[order].astype(np.int64)
        d = d[order]
        return d, fi
    except Exception:
        return None


def align_depth_nearest_fast(depth_stack: np.ndarray, frame_idx: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    """
    nearest neighbor con searchsorted (O(T log N)).
    depth_stack: [N,H,W]
    frame_idx:   [N] (sorted asc)
    query_idx:   [T]
    out: [T,H,W] float32
    """
    fi = frame_idx
    N = fi.shape[0]
    q = query_idx.astype(np.int64)

    pos = np.searchsorted(fi, q, side="left")
    pos0 = np.clip(pos - 1, 0, N - 1)
    pos1 = np.clip(pos,     0, N - 1)

    d0 = np.abs(fi[pos0] - q)
    d1 = np.abs(fi[pos1] - q)
    choose = np.where(d1 < d0, pos1, pos0)

    clip = depth_stack[choose].astype(np.float32)  # [T,H,W]
    return clip


def resize_depth_clip(depth_thw: np.ndarray, out_hw: int) -> np.ndarray:
    """
    depth_thw: [T,H,W] float32
    out: [T,out_hw,out_hw] float32
    """
    T, H, W = depth_thw.shape
    if H == out_hw and W == out_hw:
        return depth_thw
    out = np.empty((T, out_hw, out_hw), dtype=np.float32)
    for t in range(T):
        out[t] = cv2.resize(depth_thw[t], (out_hw, out_hw), interpolation=cv2.INTER_AREA)
    return out


def preprocess_depth_to_tensor(depth_thw: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    depth_thw: [T,H,W] float32 (H=W=crop)
    -> [1,1,T,H,W] float32 en device con clamp+log1p/5
    """
    d = torch.from_numpy(depth_thw).float()
    d = torch.clamp(d, 0.0, 150.0)
    d = torch.log1p(d) / 5.0
    d = d.unsqueeze(0).unsqueeze(0).contiguous()
    return d.to(device, non_blocking=True)


def load_badas_model(device: torch.device):
    # local
    try:
        from badas import BADASModel
        return BADASModel(device=str(device))
    except Exception:
        pass

    if not HAS_HF_HUB:
        raise RuntimeError("BADAS no local y no huggingface_hub.")

    loader_path = hf_hub_download(repo_id="nexar-ai/BADAS-Open", filename="badas_loader.py")
    loader_dir = os.path.dirname(loader_path)
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    from badas_loader import load_badas_model as _load
    return _load(device=str(device))


def infer_num_preds_from_badas_output(out: Any) -> Optional[int]:
    """
    Best-effort. Si BADAS devuelve lista/np.array/tensor -> len.
    Si devuelve dict con keys típicas -> len(value).
    Si no, None.
    """
    try:
        if out is None:
            return None
        if isinstance(out, (list, tuple)):
            return len(out)
        if isinstance(out, np.ndarray):
            return int(out.shape[0])
        if torch.is_tensor(out):
            return int(out.shape[0])
        if isinstance(out, dict):
            for k in ["scores", "probs", "preds", "logits", "y_score", "outputs"]:
                if k in out:
                    v = out[k]
                    if isinstance(v, (list, tuple)):
                        return len(v)
                    if isinstance(v, np.ndarray):
                        return int(v.shape[0])
                    if torch.is_tensor(v):
                        return int(v.shape[0])
        return None
    except Exception:
        return None


@dataclass
class VideoStats:
    video: str
    duration_s: float
    native_frames: int
    n_windows: int

    total_s: float
    decode_s: float
    window_build_s: float
    preproc_s: float
    depth_s: float
    forward_sys_s: float
    forward_gpu_ms: float  # sum over batches (ms)

    vram_mb: float


def _agg_mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std())


def _fmt_ms(mean_s: float, std_s: float) -> str:
    if math.isnan(mean_s):
        return "N/A"
    return f"{1000*mean_s:.1f} ± {1000*std_s:.1f}"


def _fmt_float(mean: float, std: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


# ---------------------------------------------------------------------
# 4) RUNNERS (sliding)
# ---------------------------------------------------------------------
@torch.no_grad()
def run_rgb_sliding(
    videos: List[Path],
    args,
    device: torch.device,
) -> Tuple[List[VideoStats], float]:
    """
    RGB sliding: VJEPA2TemporalBinary.
    """
    print("\n" + "=" * 80)
    print("▶ RGB SLIDING")
    print("=" * 80)

    ckpt = torch.load(args.rgb_ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) and isinstance(ckpt.get("args", {}), dict) else {}
    head_type = str(ckpt_args.get("head_type", "transformer"))

    model = _safe_construct(
        VJEPA2TemporalBinary,
        hf_repo=args.hf_repo,
        head_type=head_type,
        n_windows=1,
        unfreeze_blocks=0,
        encoder_ckpt=None,
    ).to(device).eval()

    st = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(st, strict=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    cfg = AutoConfig.from_pretrained(args.hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))

    gpu_timer = GPUTimer(device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    stats: List[VideoStats] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Warmup (1 batch)
    if videos:
        v = videos[0]
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
            T = int(video.shape[0])
            dur = T / max(fps, 1e-6)
            ends = compute_window_end_times(dur, args.hist_s, args.stride_s)
            idx0 = frame_indices_for_window(T, fps, ends[0], args.hist_s, frames_per_clip)
            clip = video[idx0].numpy()
            inputs = processor([clip], return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = model(pv)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for v in videos:
        # Total E2E
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total0 = time.perf_counter()

        # Decode
        t0 = time.perf_counter()
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            continue
        t1 = time.perf_counter()

        T = int(video.shape[0])
        if T <= 0:
            continue

        duration_s = T / max(fps, 1e-6)
        ends = compute_window_end_times(duration_s, args.hist_s, args.stride_s)
        n_windows = len(ends)

        decode_s = t1 - t0
        window_build_s = 0.0
        preproc_s = 0.0
        depth_s = 0.0
        forward_sys_s = 0.0
        forward_gpu_ms = 0.0

        # Sliding, batch
        for i in range(0, n_windows, args.batch_size):
            batch_ends = ends[i:i + args.batch_size]

            # build clips
            tb0 = time.perf_counter()
            clips = []
            for end_s in batch_ends:
                idx = frame_indices_for_window(T, fps, end_s, args.hist_s, frames_per_clip)
                clips.append(video[idx].numpy())  # [T,H,W,3]
            tb1 = time.perf_counter()
            window_build_s += (tb1 - tb0)

            # processor + H2D
            tp0 = time.perf_counter()
            inputs = processor(clips, return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            tp1 = time.perf_counter()
            preproc_s += (tp1 - tp0)

            # forward (system + gpu)
            tf0 = time.perf_counter()
            gpu_timer.start()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = model(pv)
            ms_gpu = gpu_timer.stop_ms()
            if device.type == "cuda":
                torch.cuda.synchronize()
            tf1 = time.perf_counter()

            forward_gpu_ms += ms_gpu
            forward_sys_s += (tf1 - tf0)

        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            vram_mb = float("nan")

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total1 = time.perf_counter()
        total_s = t_total1 - t_total0

        stats.append(VideoStats(
            video=v.name,
            duration_s=float(duration_s),
            native_frames=int(T),
            n_windows=int(n_windows),
            total_s=float(total_s),
            decode_s=float(decode_s),
            window_build_s=float(window_build_s),
            preproc_s=float(preproc_s),
            depth_s=float(depth_s),
            forward_sys_s=float(forward_sys_s),
            forward_gpu_ms=float(forward_gpu_ms),
            vram_mb=float(vram_mb),
        ))

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # Return peak VRAM across videos (already per-video; keep max for headline)
    max_vram = float(np.nanmax([s.vram_mb for s in stats])) if stats else float("nan")
    return stats, max_vram


@torch.no_grad()
def run_rgbd_precomputed_sliding(
    videos: List[Path],
    args,
    device: torch.device,
) -> Tuple[List[VideoStats], float]:
    """
    RGBD* sliding con depth precomputada.
    """
    print("\n" + "=" * 80)
    print("▶ RGBD* SLIDING (PRECOMPUTED DEPTH)")
    print("=" * 80)

    depth_root: Optional[Path] = Path(args.depth_root_pre) if args.depth_root_pre else None
    if depth_root is not None and not depth_root.exists():
        raise RuntimeError(f"depth_root_pre no existe: {depth_root}")
    if depth_root is None:
        print("[WARN] --depth-root-pre no provisto: usando depth=0 para todos los vídeos.")
        depth_map: Dict[str, Path] = {}
    else:
        video_root = Path(args.video_root).resolve()
        depth_map = index_depth_npz(depth_root)
        print(f"[INFO] depth npz indexados: {len(depth_map)}")

    ckpt = torch.load(args.rgbd_ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) and isinstance(ckpt.get("args", {}), dict) else {}
    depth_dim = int(ckpt_args.get("depth_dim", args.depth_dim))
    spatial_heads = int(ckpt_args.get("spatial_heads", args.spatial_heads))

    st = ckpt["model_state"] if "model_state" in ckpt else ckpt
    st_keys = list(st.keys()) if hasattr(st, "keys") else []
    use_depth_net = any(k.startswith("depth_net.") for k in st_keys)
    model_cls = VJEPA2DepthTransformerBinary_DepthNet if use_depth_net else VJEPA2DepthTransformerBinary_Patchwise

    fusion = _safe_construct(
        model_cls,
        hf_repo=args.hf_repo,
        depth_dim=depth_dim,
        spatial_heads=spatial_heads,
        unfreeze_blocks=0,
        encoder_ckpt=None,
    ).to(device).eval()

    fusion.load_state_dict(st, strict=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    cfg = AutoConfig.from_pretrained(args.hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))
    crop_size = int(getattr(cfg, "crop_size", 256))

    gpu_timer = GPUTimer(device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    stats: List[VideoStats] = []

    # Warmup
    if videos:
        v = videos[0]
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
            T = int(video.shape[0])
            dur = T / max(fps, 1e-6)
            ends = compute_window_end_times(dur, args.hist_s, args.stride_s)
            idx0 = frame_indices_for_window(T, fps, ends[0], args.hist_s, frames_per_clip)
            clip = video[idx0].numpy()
            inputs = processor([clip], return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            npz = resolve_depth_npz_for_video(depth_root, video_root, v, depth_map) if depth_root is not None else None
            dd = load_depth_npz_once(npz) if npz is not None else None
            if dd is not None:
                d_stack, d_idx = dd
                d_clip = align_depth_nearest_fast(d_stack, d_idx, idx0)
                d_clip = resize_depth_clip(d_clip, crop_size)
            else:
                d_clip = np.zeros((frames_per_clip, crop_size, crop_size), dtype=np.float32)
            d_t = preprocess_depth_to_tensor(d_clip, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = fusion(pv, d_t)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for v in videos:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total0 = time.perf_counter()

        # decode
        t0 = time.perf_counter()
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            continue
        t1 = time.perf_counter()

        T = int(video.shape[0])
        if T <= 0:
            continue
        duration_s = T / max(fps, 1e-6)
        ends = compute_window_end_times(duration_s, args.hist_s, args.stride_s)
        n_windows = len(ends)

        decode_s = t1 - t0
        window_build_s = 0.0
        preproc_s = 0.0
        depth_s = 0.0
        forward_sys_s = 0.0
        forward_gpu_ms = 0.0

        # load depth once/video
        dd0 = time.perf_counter()
        dpath = resolve_depth_npz_for_video(depth_root, video_root, v, depth_map) if depth_root is not None else None
        dd = load_depth_npz_once(dpath) if dpath is not None else None
        dd1 = time.perf_counter()
        # el tiempo de "load npz" lo metemos dentro de depth_s para ser honestos
        depth_s += (dd1 - dd0)

        if dd is None:
            # si no hay depth, rellenamos con ceros (igual que tu eval)
            d_stack = None
            d_idx = None
        else:
            d_stack, d_idx = dd

        for i in range(0, n_windows, args.batch_size):
            batch_ends = ends[i:i + args.batch_size]

            tb0 = time.perf_counter()
            clips = []
            idxs_list = []
            for end_s in batch_ends:
                idx = frame_indices_for_window(T, fps, end_s, args.hist_s, frames_per_clip)
                idxs_list.append(idx)
                clips.append(video[idx].numpy())
            tb1 = time.perf_counter()
            window_build_s += (tb1 - tb0)

            # RGB preproc
            tp0 = time.perf_counter()
            inputs = processor(clips, return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            tp1 = time.perf_counter()
            preproc_s += (tp1 - tp0)

            # Depth align + preprocess (batch)
            td0 = time.perf_counter()
            depth_batch = []
            for idx in idxs_list:
                if d_stack is None:
                    d_clip = np.zeros((frames_per_clip, crop_size, crop_size), dtype=np.float32)
                else:
                    d_clip = align_depth_nearest_fast(d_stack, d_idx, idx)   # [T,H,W] (H~224)
                    d_clip = resize_depth_clip(d_clip, crop_size)            # [T,256,256]
                depth_batch.append(d_clip)
            d_np = np.stack(depth_batch, axis=0)  # [B,T,H,W]
            # -> torch [B,1,T,H,W]
            d_t = torch.from_numpy(d_np).float()
            d_t = torch.clamp(d_t, 0.0, 150.0)
            d_t = torch.log1p(d_t) / 5.0
            d_t = d_t.unsqueeze(1).contiguous().to(device, non_blocking=True)
            td1 = time.perf_counter()
            depth_s += (td1 - td0)

            # forward
            tf0 = time.perf_counter()
            gpu_timer.start()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = fusion(pv, d_t)
            ms_gpu = gpu_timer.stop_ms()
            if device.type == "cuda":
                torch.cuda.synchronize()
            tf1 = time.perf_counter()

            forward_gpu_ms += ms_gpu
            forward_sys_s += (tf1 - tf0)

        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            vram_mb = float("nan")

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total1 = time.perf_counter()
        total_s = t_total1 - t_total0

        stats.append(VideoStats(
            video=v.name,
            duration_s=float(duration_s),
            native_frames=int(T),
            n_windows=int(n_windows),
            total_s=float(total_s),
            decode_s=float(decode_s),
            window_build_s=float(window_build_s),
            preproc_s=float(preproc_s),
            depth_s=float(depth_s),
            forward_sys_s=float(forward_sys_s),
            forward_gpu_ms=float(forward_gpu_ms),
            vram_mb=float(vram_mb),
        ))

    del fusion, processor
    torch.cuda.empty_cache()
    gc.collect()

    max_vram = float(np.nanmax([s.vram_mb for s in stats])) if stats else float("nan")
    return stats, max_vram


@torch.no_grad()
def run_midfusion_precomputed_sliding(
    videos: List[Path],
    args,
    device: torch.device,
) -> Tuple[List[VideoStats], float]:
    """
    MID-FUSION FiLM (mid_fusion.py) con depth precomputada.
    Nota: si --depth-root-pre no está, usa depth=0.
    """
    if not HAS_MIDFUSION:
        raise RuntimeError(
            "MID-FUSION no disponible (no puedo importar VJEPA2DepthMidFiLMBinary). "
            "Asegura que `scripts/depth_vjepa/mid_fusion.py` esté en el repo y ejecuta con el entorno que tenga sus deps "
            "(p.ej. `.venv/bin/python`)."
        )

    print("\n" + "=" * 80)
    print("▶ RGBD MID-FUSION (FiLM) SLIDING (PRECOMPUTED DEPTH)")
    print("=" * 80)

    depth_root: Optional[Path] = Path(args.depth_root_pre) if args.depth_root_pre else None
    if depth_root is not None and not depth_root.exists():
        raise RuntimeError(f"depth_root_pre no existe: {depth_root}")
    if depth_root is None:
        print("[WARN] --depth-root-pre no provisto: usando depth=0 para todos los vídeos.")
        depth_map: Dict[str, Path] = {}
    else:
        video_root = Path(args.video_root).resolve()
        depth_map = index_depth_npz(depth_root)
        print(f"[INFO] depth npz indexados: {len(depth_map)}")

    # Cargar checkpoint primero para poder inferir hiperparámetros (si vienen guardados).
    ckpt = torch.load(args.midfusion_ckpt, map_location="cpu")
    st = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) and isinstance(ckpt.get("args", {}), dict) else {}

    # Defaults desde checkpoint (si existen) + overrides por CLI
    depth_dim = args.midfusion_depth_dim if args.midfusion_depth_dim is not None else int(ckpt_args.get("depth_dim", args.depth_dim))
    spatial_heads = args.midfusion_spatial_heads if args.midfusion_spatial_heads is not None else int(ckpt_args.get("spatial_heads", args.spatial_heads))
    inject_layers = args.midfusion_inject_layers if args.midfusion_inject_layers is not None else str(ckpt_args.get("inject_layers", "-4,-3,-2,-1"))
    film_hidden = args.midfusion_film_hidden if args.midfusion_film_hidden is not None else int(ckpt_args.get("film_hidden", 256))
    film_scale = args.midfusion_film_scale if args.midfusion_film_scale is not None else float(ckpt_args.get("film_scale", 0.10))
    per_layer_adapters = (
        args.midfusion_per_layer_adapters
        if args.midfusion_per_layer_adapters is not None
        else bool(ckpt_args.get("film_per_layer", True))
    )
    unfreeze_blocks = args.midfusion_unfreeze_blocks if args.midfusion_unfreeze_blocks is not None else int(ckpt_args.get("unfreeze_blocks", 0))
    encoder_ckpt = args.midfusion_encoder_ckpt if args.midfusion_encoder_ckpt is not None else ckpt_args.get("encoder_ckpt", None)

    model = VJEPA2DepthMidFiLMBinary(
        hf_repo=args.hf_repo,
        depth_dim=depth_dim,
        spatial_heads=spatial_heads,
        unfreeze_blocks=unfreeze_blocks,
        encoder_ckpt=encoder_ckpt,
        inject_layers=inject_layers,
        film_hidden=film_hidden,
        film_scale=film_scale,
        per_layer_adapters=per_layer_adapters,
    ).to(device).eval()

    model.load_state_dict(st, strict=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    cfg = AutoConfig.from_pretrained(args.hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))
    crop_size = int(getattr(cfg, "crop_size", 256))

    gpu_timer = GPUTimer(device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    stats: List[VideoStats] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Warmup
    if videos:
        v = videos[0]
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
            T = int(video.shape[0])
            dur = T / max(fps, 1e-6)
            ends = compute_window_end_times(dur, args.hist_s, args.stride_s)
            idx0 = frame_indices_for_window(T, fps, ends[0], args.hist_s, frames_per_clip)
            clip = video[idx0].numpy()
            inputs = processor([clip], return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)

            npz = resolve_depth_npz_for_video(depth_root, video_root, v, depth_map) if depth_root is not None else None
            dd = load_depth_npz_once(npz) if npz is not None else None
            if dd is not None:
                d_stack, d_idx = dd
                d_clip = align_depth_nearest_fast(d_stack, d_idx, idx0)
                d_clip = resize_depth_clip(d_clip, crop_size)
            else:
                d_clip = np.zeros((frames_per_clip, crop_size, crop_size), dtype=np.float32)

            d_t = preprocess_depth_to_tensor(d_clip, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = model(pv, d_t)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for v in videos:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total0 = time.perf_counter()

        # decode
        t0 = time.perf_counter()
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            continue
        t1 = time.perf_counter()

        T = int(video.shape[0])
        if T <= 0:
            continue
        duration_s = T / max(fps, 1e-6)
        ends = compute_window_end_times(duration_s, args.hist_s, args.stride_s)
        n_windows = len(ends)

        decode_s = t1 - t0
        window_build_s = 0.0
        preproc_s = 0.0
        depth_s = 0.0
        forward_sys_s = 0.0
        forward_gpu_ms = 0.0

        # load depth once/video (opcional)
        dd0 = time.perf_counter()
        dpath = resolve_depth_npz_for_video(depth_root, video_root, v, depth_map) if depth_root is not None else None
        dd = load_depth_npz_once(dpath) if dpath is not None else None
        dd1 = time.perf_counter()
        depth_s += (dd1 - dd0)

        if dd is None:
            d_stack = None
            d_idx = None
        else:
            d_stack, d_idx = dd

        for i in range(0, n_windows, args.batch_size):
            batch_ends = ends[i:i + args.batch_size]

            tb0 = time.perf_counter()
            clips = []
            idxs_list = []
            for end_s in batch_ends:
                idx = frame_indices_for_window(T, fps, end_s, args.hist_s, frames_per_clip)
                idxs_list.append(idx)
                clips.append(video[idx].numpy())
            tb1 = time.perf_counter()
            window_build_s += (tb1 - tb0)

            # RGB preproc
            tp0 = time.perf_counter()
            inputs = processor(clips, return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            tp1 = time.perf_counter()
            preproc_s += (tp1 - tp0)

            # Depth align + preprocess (batch)
            td0 = time.perf_counter()
            depth_batch = []
            for idx in idxs_list:
                if d_stack is None:
                    d_clip = np.zeros((frames_per_clip, crop_size, crop_size), dtype=np.float32)
                else:
                    d_clip = align_depth_nearest_fast(d_stack, d_idx, idx)  # [T,H,W]
                    d_clip = resize_depth_clip(d_clip, crop_size)           # [T,256,256]
                depth_batch.append(d_clip)

            d_np = np.stack(depth_batch, axis=0)  # [B,T,H,W]
            d_t = torch.from_numpy(d_np).float()
            d_t = torch.clamp(d_t, 0.0, 150.0)
            d_t = torch.log1p(d_t) / 5.0
            d_t = d_t.unsqueeze(1).contiguous().to(device, non_blocking=True)  # [B,1,T,H,W]
            td1 = time.perf_counter()
            depth_s += (td1 - td0)

            # forward
            tf0 = time.perf_counter()
            gpu_timer.start()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = model(pv, d_t)
            ms_gpu = gpu_timer.stop_ms()
            if device.type == "cuda":
                torch.cuda.synchronize()
            tf1 = time.perf_counter()

            forward_gpu_ms += ms_gpu
            forward_sys_s += (tf1 - tf0)

        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            vram_mb = float("nan")

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total1 = time.perf_counter()
        total_s = t_total1 - t_total0

        stats.append(VideoStats(
            video=v.name,
            duration_s=float(duration_s),
            native_frames=int(T),
            n_windows=int(n_windows),
            total_s=float(total_s),
            decode_s=float(decode_s),
            window_build_s=float(window_build_s),
            preproc_s=float(preproc_s),
            depth_s=float(depth_s),
            forward_sys_s=float(forward_sys_s),
            forward_gpu_ms=float(forward_gpu_ms),
            vram_mb=float(vram_mb),
        ))

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    max_vram = float(np.nanmax([s.vram_mb for s in stats])) if stats else float("nan")
    return stats, max_vram


@torch.no_grad()
def run_rgbd_online_sliding(
    videos: List[Path],
    args,
    device: torch.device,
) -> Tuple[List[VideoStats], float]:
    """
    RGBD online: DepthAnything3 por frames únicos (no por ventana) + fusion sliding.
    """
    if not HAS_DA3:
        raise RuntimeError("DepthAnything3 no está disponible en este entorno.")

    print("\n" + "=" * 80)
    print("▶ RGBD ONLINE SLIDING (DA3 per-unique-frame)")
    print("=" * 80)

    da3 = DepthAnything3.from_pretrained(args.depth_model_name).to(device).eval()

    ckpt = torch.load(args.rgbd_ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) and isinstance(ckpt.get("args", {}), dict) else {}
    depth_dim = int(ckpt_args.get("depth_dim", args.depth_dim))
    spatial_heads = int(ckpt_args.get("spatial_heads", args.spatial_heads))

    st = ckpt["model_state"] if "model_state" in ckpt else ckpt
    st_keys = list(st.keys()) if hasattr(st, "keys") else []
    use_depth_net = any(k.startswith("depth_net.") for k in st_keys)
    model_cls = VJEPA2DepthTransformerBinary_DepthNet if use_depth_net else VJEPA2DepthTransformerBinary_Patchwise

    fusion = _safe_construct(
        model_cls,
        hf_repo=args.hf_repo,
        depth_dim=depth_dim,
        spatial_heads=spatial_heads,
        unfreeze_blocks=0,
        encoder_ckpt=None,
    ).to(device).eval()

    fusion.load_state_dict(st, strict=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    cfg = AutoConfig.from_pretrained(args.hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))
    crop_size = int(getattr(cfg, "crop_size", 256))

    gpu_timer = GPUTimer(device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    stats: List[VideoStats] = []

    # Warmup (muy pequeño)
    if videos:
        v = videos[0]
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
            T = int(video.shape[0])
            dur = T / max(fps, 1e-6)
            ends = compute_window_end_times(dur, args.hist_s, args.stride_s)
            idx0 = frame_indices_for_window(T, fps, ends[0], args.hist_s, frames_per_clip)
            # DA3 sobre esas 16 frames
            from PIL import Image
            pil = [Image.fromarray(video[i].numpy()) for i in idx0.tolist()]
            _ = da3.inference(
                image=pil,
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format=[],
            )
            # fusion warmup
            clip = video[idx0].numpy()
            inputs = processor([clip], return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            d_dummy = torch.zeros((1, 1, frames_per_clip, crop_size, crop_size), device=device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = fusion(pv, d_dummy)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    from PIL import Image

    for v in videos:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total0 = time.perf_counter()

        # decode
        t0 = time.perf_counter()
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            continue
        t1 = time.perf_counter()

        T = int(video.shape[0])
        if T <= 0:
            continue
        duration_s = T / max(fps, 1e-6)
        ends = compute_window_end_times(duration_s, args.hist_s, args.stride_s)
        n_windows = len(ends)

        decode_s = t1 - t0
        window_build_s = 0.0
        preproc_s = 0.0
        depth_s = 0.0
        forward_sys_s = 0.0
        forward_gpu_ms = 0.0

        # 1) recolectar índices únicos usados por todas las ventanas
        tu0 = time.perf_counter()
        all_idxs = []
        idxs_per_window = []
        for end_s in ends:
            idx = frame_indices_for_window(T, fps, end_s, args.hist_s, frames_per_clip)
            idxs_per_window.append(idx)
            all_idxs.append(idx)
        unique_idx = np.unique(np.concatenate(all_idxs, axis=0)).astype(np.int64)
        tu1 = time.perf_counter()
        window_build_s += (tu1 - tu0)

        # 2) depth por frames únicos (chunk)
        td0 = time.perf_counter()
        depth_map: Dict[int, np.ndarray] = {}
        # chunking
        chunk = max(1, int(args.da3_chunk))
        for i in range(0, len(unique_idx), chunk):
            ids = unique_idx[i:i + chunk].tolist()
            pil = [Image.fromarray(video[j].numpy()) for j in ids]

            # DA3 system + GPU-only (no separado aquí; esto es depth_s real)
            pred = da3.inference(
                image=pil,
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format=[],
            )
            # pred.depth es lista de mapas
            for j, d in zip(ids, pred.depth):
                d_np = np.asarray(d, dtype=np.float32)
                depth_map[int(j)] = d_np
        td1 = time.perf_counter()
        depth_s += (td1 - td0)

        # 3) sliding windows -> fusion
        for i in range(0, n_windows, args.batch_size):
            idxs_batch = idxs_per_window[i:i + args.batch_size]

            # build rgb clips
            tb0 = time.perf_counter()
            clips = []
            depth_batch = []
            for idx in idxs_batch:
                clips.append(video[idx].numpy())
                d_list = [depth_map[int(j)] for j in idx.tolist()]
                d_clip = np.stack(d_list, axis=0)  # [T,H,W]
                d_clip = resize_depth_clip(d_clip.astype(np.float32), crop_size)
                depth_batch.append(d_clip)
            tb1 = time.perf_counter()
            window_build_s += (tb1 - tb0)

            # rgb preproc
            tp0 = time.perf_counter()
            inputs = processor(clips, return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            tp1 = time.perf_counter()
            preproc_s += (tp1 - tp0)

            # depth preprocess
            td2 = time.perf_counter()
            d_np = np.stack(depth_batch, axis=0)
            d_t = torch.from_numpy(d_np).float()
            d_t = torch.clamp(d_t, 0.0, 150.0)
            d_t = torch.log1p(d_t) / 5.0
            d_t = d_t.unsqueeze(1).contiguous().to(device, non_blocking=True)
            td3 = time.perf_counter()
            depth_s += (td3 - td2)

            # forward
            tf0 = time.perf_counter()
            gpu_timer.start()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = fusion(pv, d_t)
            ms_gpu = gpu_timer.stop_ms()
            if device.type == "cuda":
                torch.cuda.synchronize()
            tf1 = time.perf_counter()

            forward_gpu_ms += ms_gpu
            forward_sys_s += (tf1 - tf0)

        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            vram_mb = float("nan")

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total1 = time.perf_counter()
        total_s = t_total1 - t_total0

        stats.append(VideoStats(
            video=v.name,
            duration_s=float(duration_s),
            native_frames=int(T),
            n_windows=int(n_windows),
            total_s=float(total_s),
            decode_s=float(decode_s),
            window_build_s=float(window_build_s),
            preproc_s=float(preproc_s),
            depth_s=float(depth_s),
            forward_sys_s=float(forward_sys_s),
            forward_gpu_ms=float(forward_gpu_ms),
            vram_mb=float(vram_mb),
        ))

    del da3, fusion, processor
    torch.cuda.empty_cache()
    gc.collect()

    max_vram = float(np.nanmax([s.vram_mb for s in stats])) if stats else float("nan")
    return stats, max_vram


@torch.no_grad()
def run_midfusion_online_sliding(
    videos: List[Path],
    args,
    device: torch.device,
) -> Tuple[List[VideoStats], float]:
    """
    MID-FUSION online: DepthAnything3 por frames únicos (no por ventana) + MID-FUSION sliding.
    """
    if not HAS_MIDFUSION:
        raise RuntimeError("MID-FUSION no disponible (no puedo importar VJEPA2DepthMidFiLMBinary).")
    if not HAS_DA3:
        raise RuntimeError("DepthAnything3 no está disponible en este entorno.")
    if not args.midfusion_ckpt:
        raise RuntimeError("Para MID-FUSION online necesitas --midfusion-ckpt.")

    print("\n" + "=" * 80)
    print("▶ RGBD MID-FUSION (FiLM) ONLINE SLIDING (DA3 per-unique-frame)")
    print("=" * 80)

    da3 = DepthAnything3.from_pretrained(args.depth_model_name).to(device).eval()

    # Cargar checkpoint para hiperparámetros.
    ckpt = torch.load(args.midfusion_ckpt, map_location="cpu")
    st = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) and isinstance(ckpt.get("args", {}), dict) else {}

    depth_dim = args.midfusion_depth_dim if args.midfusion_depth_dim is not None else int(ckpt_args.get("depth_dim", args.depth_dim))
    spatial_heads = args.midfusion_spatial_heads if args.midfusion_spatial_heads is not None else int(ckpt_args.get("spatial_heads", args.spatial_heads))
    inject_layers = args.midfusion_inject_layers if args.midfusion_inject_layers is not None else str(ckpt_args.get("inject_layers", "-4,-3,-2,-1"))
    film_hidden = args.midfusion_film_hidden if args.midfusion_film_hidden is not None else int(ckpt_args.get("film_hidden", 256))
    film_scale = args.midfusion_film_scale if args.midfusion_film_scale is not None else float(ckpt_args.get("film_scale", 0.10))
    per_layer_adapters = (
        args.midfusion_per_layer_adapters
        if args.midfusion_per_layer_adapters is not None
        else bool(ckpt_args.get("film_per_layer", True))
    )
    unfreeze_blocks = args.midfusion_unfreeze_blocks if args.midfusion_unfreeze_blocks is not None else int(ckpt_args.get("unfreeze_blocks", 0))
    encoder_ckpt = args.midfusion_encoder_ckpt if args.midfusion_encoder_ckpt is not None else ckpt_args.get("encoder_ckpt", None)

    model = VJEPA2DepthMidFiLMBinary(
        hf_repo=args.hf_repo,
        depth_dim=depth_dim,
        spatial_heads=spatial_heads,
        unfreeze_blocks=unfreeze_blocks,
        encoder_ckpt=encoder_ckpt,
        inject_layers=inject_layers,
        film_hidden=film_hidden,
        film_scale=film_scale,
        per_layer_adapters=per_layer_adapters,
    ).to(device).eval()
    model.load_state_dict(st, strict=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    cfg = AutoConfig.from_pretrained(args.hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))
    crop_size = int(getattr(cfg, "crop_size", 256))

    gpu_timer = GPUTimer(device)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    stats: List[VideoStats] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()

    from PIL import Image

    # Warmup (muy pequeño)
    if videos:
        v = videos[0]
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
            T = int(video.shape[0])
            dur = T / max(fps, 1e-6)
            ends = compute_window_end_times(dur, args.hist_s, args.stride_s)
            idx0 = frame_indices_for_window(T, fps, ends[0], args.hist_s, frames_per_clip)

            # DA3 sobre esas frames
            pil = [Image.fromarray(video[i].numpy()) for i in idx0.tolist()]
            pred = da3.inference(
                image=pil,
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format=[],
            )
            # resize aquí para ahorrar memoria
            d_clip = []
            for d in pred.depth:
                d_np = np.asarray(d, dtype=np.float32)
                d_small = cv2.resize(d_np, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                d_clip.append(d_small)
            d_clip = np.stack(d_clip, axis=0)

            # fusion warmup
            clip = video[idx0].numpy()
            inputs = processor([clip], return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            d_t = preprocess_depth_to_tensor(d_clip, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = model(pv, d_t)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for v in videos:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total0 = time.perf_counter()

        # decode
        t0 = time.perf_counter()
        try:
            video, _, info = read_video(str(v), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            continue
        t1 = time.perf_counter()

        T = int(video.shape[0])
        if T <= 0:
            continue
        duration_s = T / max(fps, 1e-6)
        ends = compute_window_end_times(duration_s, args.hist_s, args.stride_s)
        n_windows = len(ends)

        decode_s = t1 - t0
        window_build_s = 0.0
        preproc_s = 0.0
        depth_s = 0.0
        forward_sys_s = 0.0
        forward_gpu_ms = 0.0

        # 1) recolectar índices únicos usados por todas las ventanas
        tu0 = time.perf_counter()
        all_idxs = []
        idxs_per_window = []
        for end_s in ends:
            idx = frame_indices_for_window(T, fps, end_s, args.hist_s, frames_per_clip)
            idxs_per_window.append(idx)
            all_idxs.append(idx)
        unique_idx = np.unique(np.concatenate(all_idxs, axis=0)).astype(np.int64)
        tu1 = time.perf_counter()
        window_build_s += (tu1 - tu0)

        # 2) depth por frames únicos (chunk)
        td0 = time.perf_counter()
        depth_map: Dict[int, np.ndarray] = {}
        chunk = max(1, int(args.da3_chunk))
        for i in range(0, len(unique_idx), chunk):
            ids = unique_idx[i:i + chunk].tolist()
            pil = [Image.fromarray(video[j].numpy()) for j in ids]
            pred = da3.inference(
                image=pil,
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format=[],
            )
            for j, d in zip(ids, pred.depth):
                d_np = np.asarray(d, dtype=np.float32)
                d_small = cv2.resize(d_np, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                depth_map[int(j)] = d_small
        td1 = time.perf_counter()
        depth_s += (td1 - td0)

        # 3) sliding windows -> fusion
        for i in range(0, n_windows, args.batch_size):
            idxs_batch = idxs_per_window[i:i + args.batch_size]

            # build rgb clips + depth clips
            tb0 = time.perf_counter()
            clips = []
            depth_batch = []
            for idx in idxs_batch:
                clips.append(video[idx].numpy())
                d_list = [depth_map[int(j)] for j in idx.tolist()]
                d_clip = np.stack(d_list, axis=0)  # [T,H,W] ya resizeado
                depth_batch.append(d_clip)
            tb1 = time.perf_counter()
            window_build_s += (tb1 - tb0)

            # rgb preproc
            tp0 = time.perf_counter()
            inputs = processor(clips, return_tensors="pt")
            pv = inputs["pixel_values_videos"].to(device, non_blocking=True)
            tp1 = time.perf_counter()
            preproc_s += (tp1 - tp0)

            # depth preprocess
            td2 = time.perf_counter()
            d_np = np.stack(depth_batch, axis=0)  # [B,T,H,W]
            d_t = torch.from_numpy(d_np).float()
            d_t = torch.clamp(d_t, 0.0, 150.0)
            d_t = torch.log1p(d_t) / 5.0
            d_t = d_t.unsqueeze(1).contiguous().to(device, non_blocking=True)  # [B,1,T,H,W]
            td3 = time.perf_counter()
            depth_s += (td3 - td2)

            # forward
            tf0 = time.perf_counter()
            gpu_timer.start()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = model(pv, d_t)
            ms_gpu = gpu_timer.stop_ms()
            if device.type == "cuda":
                torch.cuda.synchronize()
            tf1 = time.perf_counter()

            forward_gpu_ms += ms_gpu
            forward_sys_s += (tf1 - tf0)

        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            vram_mb = float("nan")

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total1 = time.perf_counter()
        total_s = t_total1 - t_total0

        stats.append(VideoStats(
            video=v.name,
            duration_s=float(duration_s),
            native_frames=int(T),
            n_windows=int(n_windows),
            total_s=float(total_s),
            decode_s=float(decode_s),
            window_build_s=float(window_build_s),
            preproc_s=float(preproc_s),
            depth_s=float(depth_s),
            forward_sys_s=float(forward_sys_s),
            forward_gpu_ms=float(forward_gpu_ms),
            vram_mb=float(vram_mb),
        ))

    del da3, model, processor
    torch.cuda.empty_cache()
    gc.collect()

    max_vram = float(np.nanmax([s.vram_mb for s in stats])) if stats else float("nan")
    return stats, max_vram


def run_badas_e2e(
    videos: List[Path],
    args,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    BADAS e2e: predict(path).
    Devuelve lista de dicts por vídeo con:
      total_s, duration_s, native_frames, n_preds(optional)
    """
    print("\n" + "=" * 80)
    print("▶ BADAS-OPEN E2E (black box)")
    print("=" * 80)

    try:
        badas = load_badas_model(device)
    except Exception as e:
        print(f"[BADAS] No se pudo cargar: {e}")
        return [], float("nan")

    recs: List[Dict[str, Any]] = []

    # warmup
    if videos:
        try:
            _ = badas.predict(str(videos[0]))
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for v in videos:
        # get native frames + duration (cv2 rápido)
        cap = cv2.VideoCapture(str(v))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        duration_s = n_frames / max(fps, 1e-6) if n_frames > 0 else float("nan")
        if n_frames <= 0:
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = badas.predict(str(v))
        except Exception:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        n_preds = infer_num_preds_from_badas_output(out)

        recs.append({
            "video": v.name,
            "duration_s": float(duration_s),
            "native_frames": int(n_frames),
            "total_s": float(t1 - t0),
            "n_preds": int(n_preds) if n_preds is not None else None,
        })

    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        vram_mb = float("nan")

    del badas
    torch.cuda.empty_cache()
    gc.collect()

    return recs, float(vram_mb)


# ---------------------------------------------------------------------
# 5) REPORTING
# ---------------------------------------------------------------------
def summarize_pipeline(name: str, stats: List[VideoStats], frames_per_clip: int) -> Dict[str, Any]:
    if not stats:
        return {"name": name, "n": 0}

    total = [s.total_s for s in stats]
    decode = [s.decode_s for s in stats]
    winb = [s.window_build_s for s in stats]
    prep = [s.preproc_s for s in stats]
    depth = [s.depth_s for s in stats]
    fws = [s.forward_sys_s for s in stats]
    vram = [s.vram_mb for s in stats]
    dur = [s.duration_s for s in stats]
    nwin = [s.n_windows for s in stats]
    nframes = [s.native_frames for s in stats]
    gpu_ms = [s.forward_gpu_ms for s in stats]  # sum per video

    # Derived
    # xRT = duration / total
    xrt = [ (d / t) if (t > 0 and d > 0) else float("nan") for d, t in zip(dur, total) ]
    # ms/pred (system)
    ms_per_pred = [ (1000.0 * t / w) if (w > 0) else float("nan") for t, w in zip(total, nwin) ]
    # pred/s
    pred_per_s = [ (w / t) if (t > 0) else float("nan") for t, w in zip(total, nwin) ]
    # throughput native fps
    native_fps = [ (nf / t) if (t > 0) else float("nan") for nf, t in zip(nframes, total) ]
    # effective processed fps = (nwin * frames_per_clip)/t
    eff_fps = [ ((w * frames_per_clip) / t) if (t > 0) else float("nan") for w, t in zip(nwin, total) ]
    # GPU-only ms/pred = (gpu_ms_sum / nwin)
    gpu_ms_per_pred = [ (gm / w) if (w > 0) else float("nan") for gm, w in zip(gpu_ms, nwin) ]

    out = {
        "name": name,
        "n": len(stats),
        "total_ms_mean_std": _agg_mean_std(total),
        "decode_ms_mean_std": _agg_mean_std(decode),
        "winbuild_ms_mean_std": _agg_mean_std(winb),
        "preproc_ms_mean_std": _agg_mean_std(prep),
        "depth_ms_mean_std": _agg_mean_std(depth),
        "forward_sys_ms_mean_std": _agg_mean_std(fws),
        "xrt_mean_std": _agg_mean_std(xrt),
        "ms_per_pred_mean_std": _agg_mean_std(ms_per_pred),
        "pred_per_s_mean_std": _agg_mean_std(pred_per_s),
        "native_fps_mean_std": _agg_mean_std(native_fps),
        "eff_fps_mean_std": _agg_mean_std(eff_fps),
        "gpu_ms_per_pred_mean_std": _agg_mean_std(gpu_ms_per_pred),
        "vram_mb_max": float(np.nanmax(vram)),
        "frames_per_clip": int(frames_per_clip),
    }
    return out


def summarize_badas(name: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {"name": name, "n": 0}

    total = [r["total_s"] for r in recs]
    dur = [r["duration_s"] for r in recs]
    nframes = [r["native_frames"] for r in recs]
    xrt = [ (d / t) if (t > 0 and d > 0) else float("nan") for d, t in zip(dur, total) ]
    native_fps = [ (nf / t) if (t > 0) else float("nan") for nf, t in zip(nframes, total) ]

    # pred stats if available
    preds = [r.get("n_preds", None) for r in recs]
    preds_ok = [p for p in preds if isinstance(p, int) and p > 0]
    ms_per_pred = None
    pred_per_s = None
    if preds_ok:
        # align with recs where n_preds ok
        ms_per_pred_list = []
        pred_per_s_list = []
        for r in recs:
            p = r.get("n_preds", None)
            if isinstance(p, int) and p > 0:
                t = r["total_s"]
                ms_per_pred_list.append(1000.0 * t / p)
                pred_per_s_list.append(p / t if t > 0 else float("nan"))
        ms_per_pred = _agg_mean_std(ms_per_pred_list)
        pred_per_s = _agg_mean_std(pred_per_s_list)

    return {
        "name": name,
        "n": len(recs),
        "total_ms_mean_std": _agg_mean_std(total),
        "xrt_mean_std": _agg_mean_std(xrt),
        "native_fps_mean_std": _agg_mean_std(native_fps),
        "ms_per_pred_mean_std": ms_per_pred,   # puede ser None
        "pred_per_s_mean_std": pred_per_s,     # puede ser None
    }


def print_summary_table(summaries: List[Dict[str, Any]]):
    rows = [["Pipeline", "E2E [ms/video]", "xRealTime", "ms/pred (sys)", "pred/s", "native FPS", "eff FPS (win*16)", "GPU ms/pred", "VRAM max [MiB]"]]
    for s in summaries:
        if s.get("n", 0) == 0:
            continue

        # pipeline with breakdown
        t_m, t_s = s["total_ms_mean_std"]
        xr_m, xr_s = s.get("xrt_mean_std", (float("nan"), float("nan")))
        mp_m, mp_s = s.get("ms_per_pred_mean_std", (float("nan"), float("nan")))
        ps_m, ps_s = s.get("pred_per_s_mean_std", (float("nan"), float("nan")))
        nf_m, nf_s = s.get("native_fps_mean_std", (float("nan"), float("nan")))
        ef_m, ef_s = s.get("eff_fps_mean_std", (float("nan"), float("nan")))
        gp_m, gp_s = s.get("gpu_ms_per_pred_mean_std", (float("nan"), float("nan")))
        vram = s.get("vram_mb_max", float("nan"))

        rows.append([
            s["name"],
            _fmt_ms(t_m, t_s),
            _fmt_float(xr_m, xr_s, 3),
            _fmt_float(mp_m, mp_s, 2),
            _fmt_float(ps_m, ps_s, 3),
            _fmt_float(nf_m, nf_s, 1),
            _fmt_float(ef_m, ef_s, 1),
            _fmt_float(gp_m, gp_s, 2),
            f"{vram:.1f}" if not math.isnan(vram) else "N/A",
        ])

    print("\n" + "=" * 110)
    print("📊 SUMMARY (Setting B: sliding anytime)")
    print("=" * 110)
    print(tabulate(rows, headers="firstrow", tablefmt="fancy_grid"))


def print_breakdown_table(name: str, summary: Dict[str, Any]):
    if summary.get("n", 0) == 0:
        return
    rows = [["Component", "mean ± std (ms/video)", "share (%)"]]

    t_total_m, t_total_s = summary["total_ms_mean_std"]
    comps = [
        ("decode", summary["decode_ms_mean_std"]),
        ("window_build", summary["winbuild_ms_mean_std"]),
        ("preproc_rgb", summary["preproc_ms_mean_std"]),
        ("depth_io+prep", summary["depth_ms_mean_std"]),
        ("forward_sys", summary["forward_sys_ms_mean_std"]),
    ]
    for cname, (m, s) in comps:
        share = (m / t_total_m * 100.0) if (not math.isnan(m) and not math.isnan(t_total_m) and t_total_m > 0) else float("nan")
        rows.append([cname, _fmt_ms(m, s), f"{share:.1f}" if not math.isnan(share) else "N/A"])

    print("\n" + "-" * 110)
    print(f"🔍 BREAKDOWN: {name}")
    print("-" * 110)
    print(tabulate(rows, headers="firstrow", tablefmt="fancy_grid"))


# ---------------------------------------------------------------------
# 6) MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--video-root", type=str, required=True)
    ap.add_argument("--hf-repo", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2")

    # Puedes pasar múltiples separadas por coma o repitiendo el flag.
    ap.add_argument("--rgb-ckpt", type=str, action="append", default=None)
    ap.add_argument("--rgbd-ckpt", type=str, action="append", default=None)
    ap.add_argument("--midfusion-ckpt", type=str, action="append", default=None,
                    help="Checkpoint(s) de MID-FUSION FiLM (p.ej. best_midfilm_*.pt)")
    ap.add_argument("--depth-root-pre", type=str, default=None)

    ap.add_argument("--include-rgbd-online", action="store_true")
    ap.add_argument("--include-midfusion-online", action="store_true")
    ap.add_argument("--depth-model-name", type=str, default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--da3-chunk", type=int, default=16, help="batch de frames únicos para DA3 online")

    ap.add_argument("--include-badas", action="store_true")

    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--stride-s", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-videos", type=int, default=30)

    ap.add_argument("--depth-dim", type=int, default=128, help="fallback si el ckpt no trae args.depth_dim")
    ap.add_argument("--spatial-heads", type=int, default=4, help="fallback si el ckpt no trae args.spatial_heads")

    # Mid-fusion overrides (por defecto intenta leerlos del checkpoint si existen)
    ap.add_argument("--midfusion-depth-dim", type=int, default=None)
    ap.add_argument("--midfusion-spatial-heads", type=int, default=None)
    ap.add_argument("--midfusion-inject-layers", type=str, default=None)
    ap.add_argument("--midfusion-film-hidden", type=int, default=None)
    ap.add_argument("--midfusion-film-scale", type=float, default=None)
    ap.add_argument("--midfusion-per-layer-adapters", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--midfusion-unfreeze-blocks", type=int, default=None)
    ap.add_argument("--midfusion-encoder-ckpt", type=str, default=None)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true", help="autocast para tus modelos (recomendado en GPU)")
    ap.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--save-json", type=str, default=None, help="guarda resultados agregados en JSON")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    rgb_ckpts = _normalize_ckpt_list(args.rgb_ckpt)
    rgbd_ckpts = _normalize_ckpt_list(args.rgbd_ckpt)
    mid_ckpts = _normalize_ckpt_list(args.midfusion_ckpt)

    if not any([rgb_ckpts, rgbd_ckpts, mid_ckpts, args.include_rgbd_online, args.include_midfusion_online, args.include_badas]):
        raise RuntimeError("No hay nada que ejecutar: pasa --rgb-ckpt y/o --rgbd-ckpt y/o --midfusion-ckpt y/o --include-badas.")

    video_root = Path(args.video_root).resolve()
    if not video_root.exists():
        raise RuntimeError(f"video-root no existe: {video_root}")

    videos = list_videos(video_root, args.max_videos)
    if not videos:
        raise RuntimeError("No hay vídeos.")

    # Config: frames_per_clip
    cfg = AutoConfig.from_pretrained(args.hf_repo)
    frames_per_clip = int(getattr(cfg, "frames_per_clip", 16))

    all_summaries: List[Dict[str, Any]] = []
    breakdowns: List[Dict[str, Any]] = []

    # 1) RGB sliding
    rgb_stats_all: Dict[str, List[VideoStats]] = {}
    if rgb_ckpts:
        for ckpt_path in rgb_ckpts:
            a = argparse.Namespace(**vars(args))
            a.rgb_ckpt = ckpt_path
            name = f"RGB sliding ({_short_ckpt_name(ckpt_path)})"
            st, _ = run_rgb_sliding(videos, a, device)
            rgb_stats_all[name] = st
            s = summarize_pipeline(name, st, frames_per_clip)
            all_summaries.append(s)
            breakdowns.append(s)
    else:
        print("[INFO] RGB no ejecutado (falta --rgb-ckpt).")

    # 2) RGBD* precomputed sliding (si procede)
    rgbd_pre_stats_all: Dict[str, List[VideoStats]] = {}
    if rgbd_ckpts and args.depth_root_pre:
        for ckpt_path in rgbd_ckpts:
            a = argparse.Namespace(**vars(args))
            a.rgbd_ckpt = ckpt_path
            name = f"RGBD* precomp sliding ({_short_ckpt_name(ckpt_path)})"
            st, _ = run_rgbd_precomputed_sliding(videos, a, device)
            rgbd_pre_stats_all[name] = st
            s = summarize_pipeline(name, st, frames_per_clip)
            all_summaries.append(s)
            breakdowns.append(s)
    elif rgbd_ckpts and not args.depth_root_pre:
        print("[INFO] RGBD* precomputed no ejecutado (falta --depth-root-pre).")
    else:
        print("[INFO] RGBD* no ejecutado (falta --rgbd-ckpt).")

    # 2.5) MID-FUSION FiLM sliding
    mid_pre_stats_all: Dict[str, List[VideoStats]] = {}
    if mid_ckpts and args.depth_root_pre:
        for ckpt_path in mid_ckpts:
            a = argparse.Namespace(**vars(args))
            a.midfusion_ckpt = ckpt_path
            name = f"RGBD MID-FUSION precomp sliding ({_short_ckpt_name(ckpt_path)})"
            st, _ = run_midfusion_precomputed_sliding(videos, a, device)
            mid_pre_stats_all[name] = st
            s = summarize_pipeline(name, st, frames_per_clip)
            all_summaries.append(s)
            breakdowns.append(s)
    elif mid_ckpts and not args.depth_root_pre:
        print("[INFO] MID-FUSION precomputed no ejecutado (falta --depth-root-pre).")
    else:
        print("[INFO] MID-FUSION no ejecutado (falta --midfusion-ckpt).")

    # 3) RGBD online sliding (opcional)
    if args.include_rgbd_online:
        if not rgbd_ckpts:
            raise RuntimeError("Para RGBD online necesitas --rgbd-ckpt (fusion weights).")
        for ckpt_path in rgbd_ckpts:
            a = argparse.Namespace(**vars(args))
            a.rgbd_ckpt = ckpt_path
            name = f"RGBD online sliding ({_short_ckpt_name(ckpt_path)})"
            rgbd_on_stats, _ = run_rgbd_online_sliding(videos, a, device)
            s = summarize_pipeline(name, rgbd_on_stats, frames_per_clip)
            all_summaries.append(s)
            breakdowns.append(s)

    # 3.5) MID-FUSION online sliding (opcional)
    if args.include_midfusion_online:
        if not mid_ckpts:
            raise RuntimeError("Para MID-FUSION online necesitas --midfusion-ckpt.")
        for ckpt_path in mid_ckpts:
            a = argparse.Namespace(**vars(args))
            a.midfusion_ckpt = ckpt_path
            name = f"RGBD MID-FUSION online sliding ({_short_ckpt_name(ckpt_path)})"
            mid_on_stats, _ = run_midfusion_online_sliding(videos, a, device)
            s = summarize_pipeline(name, mid_on_stats, frames_per_clip)
            all_summaries.append(s)
            breakdowns.append(s)

    # 4) BADAS e2e (opcional)
    badas_rec = []
    badas_vram = float("nan")
    if args.include_badas:
        badas_rec, badas_vram = run_badas_e2e(videos, args, device)
        bsum = summarize_badas("BADAS-Open (black-box)", badas_rec)
        # adaptar a formato tabla
        if bsum.get("n", 0) > 0:
            t_m, t_s = bsum["total_ms_mean_std"]
            xr_m, xr_s = bsum["xrt_mean_std"]
            nf_m, nf_s = bsum["native_fps_mean_std"]
            mp = bsum.get("ms_per_pred_mean_std", None)
            ps = bsum.get("pred_per_s_mean_std", None)
            out = {
                "name": bsum["name"],
                "n": bsum["n"],
                "total_ms_mean_std": (t_m, t_s),
                "xrt_mean_std": (xr_m, xr_s),
                "ms_per_pred_mean_std": mp if mp is not None else (float("nan"), float("nan")),
                "pred_per_s_mean_std": ps if ps is not None else (float("nan"), float("nan")),
                "native_fps_mean_std": (nf_m, nf_s),
                "eff_fps_mean_std": (float("nan"), float("nan")),
                "gpu_ms_per_pred_mean_std": (float("nan"), float("nan")),
                "vram_mb_max": badas_vram,
            }
            all_summaries.append(out)

    # Print
    print_summary_table(all_summaries)
    for s in breakdowns:
        print_breakdown_table(s["name"], s)

    # Save JSON (agregados + per-video opcional)
    if args.save_json:
        payload = {
            "args": vars(args),
            "frames_per_clip": frames_per_clip,
            "summaries": all_summaries,
            "per_video": {},
        }
        for name, recs in rgb_stats_all.items():
            payload["per_video"][name] = [asdict(x) for x in recs]
        for name, recs in rgbd_pre_stats_all.items():
            payload["per_video"][name] = [asdict(x) for x in recs]
        for name, recs in mid_pre_stats_all.items():
            payload["per_video"][name] = [asdict(x) for x in recs]
        if args.include_badas:
            payload["badas_per_video"] = badas_rec

        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2))
        print(f"[JSON] guardado en: {outp}")


if __name__ == "__main__":
    main()
