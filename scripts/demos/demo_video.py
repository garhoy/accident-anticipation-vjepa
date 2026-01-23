#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_video.py

Stream a video and print/overlay accident probability in near real time.

Supports:
  - RGB checkpoints (VJEPA2TemporalBinary)
  - Depth checkpoints (DepthNet / Patchwise / Mid-Fusion) with precomputed depth

Example:
  PYTHONPATH=scripts/vjepa_core .venv/bin/python scripts/demos/demo_video.py \
    --video 00023.mp4 \
    --csv data/metadata/Nexar/train.csv \
    --checkpoint checkpoints/depth_vjepa/checkpoints_vjepa_tf_rgb_frozen_hn05_j0-2_pw3/best_robust_model.pt \
    --model-type auto \
    --head-type transformer
"""

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoVideoProcessor, VJEPA2Model


# ---------------------------------------------------------------------
# Paths (robust to repo layout)
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()


def _find_repo_root(start: Path) -> Path:
    for p in (start,) + tuple(start.parents):
        if (p / "pyproject.toml").exists() or (p / "requirements.txt").exists() or (p / ".git").exists():
            return p
    return start.parents[2] if len(start.parents) >= 3 else start.parent


REPO_ROOT = _find_repo_root(CURRENT_FILE)
SCRIPTS_DIR = REPO_ROOT / "scripts"
VJEPA_CORE_DIR = SCRIPTS_DIR / "vjepa_core"
DEPTH_VJEPA_DIR = SCRIPTS_DIR / "depth_vjepa"

for p in (REPO_ROOT, SCRIPTS_DIR, VJEPA_CORE_DIR, DEPTH_VJEPA_DIR, REPO_ROOT / "src"):
    if p.exists():
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


# ---------------------------------------------------------------------
# Heads for RGB model
# ---------------------------------------------------------------------
try:
    from models import (
        BADASAttentiveProbe,
        DeformableEventQueryHead,
        TransformerPerFrame,
    )
except Exception:
    print("[WARN] models.py not found. Run from repo root or set PYTHONPATH.")


# ---------------------------------------------------------------------
# Depth models (imported lazily)
# ---------------------------------------------------------------------
VJEPA2DepthTransformerBinary_DepthNet = None
VJEPA2DepthTransformerBinary_Patchwise = None
VJEPA2DepthMidFiLMBinary = None


def _import_depth_models() -> None:
    global VJEPA2DepthTransformerBinary_DepthNet
    global VJEPA2DepthTransformerBinary_Patchwise
    global VJEPA2DepthMidFiLMBinary

    if VJEPA2DepthTransformerBinary_DepthNet is not None:
        return

    # DepthNet (per-frame depth)
    try:
        from train_nexar_depth_general import (
            VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet,  # type: ignore
        )
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) == "wandb":
            import types
            sys.modules["wandb"] = types.SimpleNamespace(
                init=lambda *a, **k: None,
                log=lambda *a, **k: None,
                finish=lambda *a, **k: None,
            )
            from train_nexar_depth_general import (
                VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet,  # type: ignore
            )
        else:
            from eval_nexar_fusion_depth import (
                VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet,  # type: ignore
            )
    except Exception:
        from eval_nexar_fusion_depth import (
            VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_DepthNet,  # type: ignore
        )

    # Patchwise / DenseFusion
    try:
        from train_nexar_patchwise import (
            VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise,  # type: ignore
        )
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) == "wandb":
            import types
            sys.modules["wandb"] = types.SimpleNamespace(
                init=lambda *a, **k: None,
                log=lambda *a, **k: None,
                finish=lambda *a, **k: None,
            )
            from train_nexar_patchwise import (
                VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise,  # type: ignore
            )
        else:
            from eval_nexar_fusion_depth_patchwise import (
                VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise,  # type: ignore
            )
    except Exception:
        from eval_nexar_fusion_depth_patchwise import (
            VJEPA2DepthTransformerBinary as VJEPA2DepthTransformerBinary_Patchwise,  # type: ignore
        )

    # Mid-Fusion (FiLM)
    try:
        from mid_fusion import VJEPA2DepthMidFiLMBinary  # type: ignore
    except Exception:
        VJEPA2DepthMidFiLMBinary = None

    globals()["VJEPA2DepthTransformerBinary_DepthNet"] = VJEPA2DepthTransformerBinary_DepthNet
    globals()["VJEPA2DepthTransformerBinary_Patchwise"] = VJEPA2DepthTransformerBinary_Patchwise
    globals()["VJEPA2DepthMidFiLMBinary"] = VJEPA2DepthMidFiLMBinary


# ---------------------------------------------------------------------
# RGB model (minimal)
# ---------------------------------------------------------------------
class VJEPA2TemporalBinary(nn.Module):
    def __init__(self, hf_repo: str, head_type: str = "badas_attn"):
        super().__init__()
        self.head_type = head_type
        self.vjepa2 = VJEPA2Model.from_pretrained(hf_repo)
        cfg = self.vjepa2.config
        self.embed_dim = cfg.hidden_size
        self.frames_per_clip = getattr(cfg, "frames_per_clip", 16)
        self.grid = getattr(cfg, "crop_size", 256) // getattr(cfg, "patch_size", 16)
        self.tubelet = cfg.tubelet_size
        self.steps = self.frames_per_clip // self.tubelet
        self._init_head()

    def _init_head(self):
        if self.head_type == "badas_attn":
            self.head = BADASAttentiveProbe(self.embed_dim, 1, self.grid, self.grid, self.frames_per_clip, self.tubelet)
        elif self.head_type == "deformable_event":
            self.head = DeformableEventQueryHead(self.embed_dim, 1, self.grid, self.grid, self.frames_per_clip, self.tubelet)
        else:
            self.head = TransformerPerFrame(self.embed_dim, 1, d_model=384)

    def forward(self, pixel_values_videos, **kwargs):
        out = self.vjepa2(pixel_values_videos=pixel_values_videos, output_hidden_states=False)
        tokens = out.last_hidden_state

        if self.head_type in ["badas_attn", "deformable_event"]:
            logits_t = self.head(tokens)
        else:
            B, L, D = tokens.shape
            if L != (self.steps * self.grid**2):
                tokens = tokens[:, 1:, :]
            seq = tokens.view(B, self.steps, self.grid**2, D).mean(dim=2)
            mask = torch.ones(seq.shape[0], seq.shape[1], dtype=torch.bool, device=seq.device)
            logits_seq = self.head(seq, mask=mask)
            logits_t = nn.functional.interpolate(
                logits_seq.transpose(1, 2), size=self.frames_per_clip, mode="linear"
            ).transpose(1, 2)

        return logits_t.max(dim=1).values.squeeze(-1)


# ---------------------------------------------------------------------
# Depth utilities (copied from speed_test.py)
# ---------------------------------------------------------------------
def index_depth_npz(depth_root: Path) -> Dict[str, Path]:
    depth_map: Dict[str, Path] = {}
    for p in depth_root.rglob("*.npz"):
        if p.stem not in depth_map:
            depth_map[p.stem] = p
    return depth_map


def resolve_depth_npz_for_video(
    depth_root: Path,
    video_root: Path,
    video_path: Path,
    depth_index: Optional[Dict[str, Path]] = None,
) -> Optional[Path]:
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


def load_depth_npz_once(npz_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        data = np.load(str(npz_path))
        d = data["depth"]
        fi = data["frame_idx"]
        if d.ndim != 3 or fi.ndim != 1 or d.shape[0] != fi.shape[0] or d.shape[0] == 0:
            return None
        order = np.argsort(fi)
        fi = fi[order].astype(np.int64)
        d = d[order]
        return d, fi
    except Exception:
        return None


def align_depth_nearest_fast(depth_stack: np.ndarray, frame_idx: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    fi = frame_idx
    N = fi.shape[0]
    q = query_idx.astype(np.int64)

    pos = np.searchsorted(fi, q, side="left")
    pos0 = np.clip(pos - 1, 0, N - 1)
    pos1 = np.clip(pos, 0, N - 1)

    d0 = np.abs(fi[pos0] - q)
    d1 = np.abs(fi[pos1] - q)
    choose = np.where(d1 < d0, pos1, pos0)

    clip = depth_stack[choose].astype(np.float32)
    return clip


def resize_depth_clip(depth_thw: np.ndarray, out_hw: int) -> np.ndarray:
    T, H, W = depth_thw.shape
    if H == out_hw and W == out_hw:
        return depth_thw
    out = np.empty((T, out_hw, out_hw), dtype=np.float32)
    for t in range(T):
        out[t] = cv2.resize(depth_thw[t], (out_hw, out_hw), interpolation=cv2.INTER_AREA)
    return out


def preprocess_depth_to_tensor(depth_thw: np.ndarray, device: torch.device) -> torch.Tensor:
    d = torch.from_numpy(depth_thw).float()
    d = torch.clamp(d, 0.0, 150.0)
    d = torch.log1p(d) / 5.0
    d = d.unsqueeze(0).unsqueeze(0).contiguous()
    return d.to(device, non_blocking=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def lookup_time_of_event(csv_path: Optional[Path], vid_id: str) -> Optional[float]:
    if not csv_path or not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("id", "").strip() == vid_id:
                return parse_float(row.get("time_of_event", ""))
    return None


def infer_model_type(state_dict: Dict[str, torch.Tensor], override: str) -> str:
    if override != "auto":
        return override
    keys = list(state_dict.keys())
    if any(k.startswith("film.") for k in keys):
        return "midfusion"
    if any(k.startswith("depth_net.") for k in keys):
        return "depth_net"
    if any(k.startswith("depth_tokenizer.") for k in keys) or any(k.startswith("fusion.") for k in keys):
        return "patchwise"
    return "rgb"


def pick_arg(args_dict: Dict, key: str, fallback):
    if args_dict is None:
        return fallback
    v = args_dict.get(key, None)
    return v if v is not None else fallback


def draw_overlay(frame_bgr: np.ndarray, prob: Optional[float], t_sec: float, t_event: Optional[float]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    bar_h = 45
    out = frame_bgr.copy()
    cv2.rectangle(out, (0, h - bar_h), (w, h), (0, 0, 0), -1)

    if prob is None:
        text = f"t={t_sec:6.2f}s  prob=--"
    else:
        text = f"t={t_sec:6.2f}s  prob={prob:0.3f}"
    if t_event is not None:
        text += f"  event={t_event:0.2f}s"

    cv2.putText(out, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if prob is not None:
        bar_w = int((w - 20) * max(0.0, min(1.0, prob)))
        cv2.rectangle(out, (10, h - bar_h + 8), (w - 10, h - bar_h + 20), (200, 200, 200), 1)
        cv2.rectangle(out, (10, h - bar_h + 8), (10 + bar_w, h - bar_h + 20), (0, 0, 255), -1)

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Path to video file.")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV with time_of_event.")
    ap.add_argument("--model-type", type=str, default="auto", choices=["auto", "rgb", "depth_net", "patchwise", "midfusion"])
    ap.add_argument("--hf-repo", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--head-type", type=str, default="transformer")
    ap.add_argument("--hist-s", type=float, default=5.0, help="History seconds for each clip.")
    ap.add_argument("--stride-s", type=float, default=0.2, help="Seconds between predictions.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--no-display", action="store_true", help="Do not open a display window.")
    ap.add_argument("--save-csv", type=str, default=None, help="Optional output CSV with time/prob.")
    ap.add_argument("--depth-root", type=str, default=None, help="Root with precomputed depth .npz files.")
    ap.add_argument("--depth-npz", type=str, default=None, help="Direct .npz path for this video.")
    ap.add_argument("--depth-dim", type=int, default=128, help="Fallback depth_dim if ckpt args missing.")
    ap.add_argument("--midfusion-depth-dim", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds (debug).")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        ckpt_args = ckpt.get("args", {}) or {}
    else:
        state = ckpt
        ckpt_args = {}

    model_type = infer_model_type(state, args.model_type)
    hf_repo = pick_arg(ckpt_args, "hf_repo", args.hf_repo)

    # Build model
    if model_type == "rgb":
        head_type = pick_arg(ckpt_args, "head_type", args.head_type)
        model = VJEPA2TemporalBinary(hf_repo, head_type=head_type)
    else:
        _import_depth_models()
        depth_dim = pick_arg(ckpt_args, "depth_dim", args.depth_dim)
        spatial_heads = pick_arg(ckpt_args, "spatial_heads", 1)
        encoder_ckpt = pick_arg(ckpt_args, "encoder_ckpt", None)

        if model_type == "depth_net":
            model = VJEPA2DepthTransformerBinary_DepthNet(  # type: ignore
                hf_repo, depth_dim=depth_dim, unfreeze_blocks=0, encoder_ckpt=encoder_ckpt
            )
        elif model_type == "patchwise":
            model = VJEPA2DepthTransformerBinary_Patchwise(  # type: ignore
                hf_repo, depth_dim=depth_dim, spatial_heads=spatial_heads, unfreeze_blocks=0, encoder_ckpt=encoder_ckpt
            )
        elif model_type == "midfusion":
            if VJEPA2DepthMidFiLMBinary is None:
                raise RuntimeError("Mid-fusion model not available in this repo/env.")
            mf_depth_dim = args.midfusion_depth_dim if args.midfusion_depth_dim is not None else depth_dim
            inject_layers = pick_arg(ckpt_args, "inject_layers", "-4,-3,-2,-1")
            film_hidden = pick_arg(ckpt_args, "film_hidden", 256)
            film_scale = pick_arg(ckpt_args, "film_scale", 0.10)
            per_layer_adapters = pick_arg(ckpt_args, "per_layer_adapters", True)
            model = VJEPA2DepthMidFiLMBinary(  # type: ignore
                hf_repo,
                depth_dim=mf_depth_dim,
                spatial_heads=spatial_heads,
                unfreeze_blocks=0,
                encoder_ckpt=encoder_ckpt,
                inject_layers=inject_layers,
                film_hidden=film_hidden,
                film_scale=film_scale,
                per_layer_adapters=per_layer_adapters,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    t_event = lookup_time_of_event(Path(args.csv) if args.csv else None, video_path.stem)

    # Depth setup (optional)
    depth_stack = None
    depth_idx = None
    if model_type != "rgb":
        if args.depth_npz:
            npz_path = Path(args.depth_npz).expanduser().resolve()
            if npz_path.exists():
                dd = load_depth_npz_once(npz_path)
                if dd:
                    depth_stack, depth_idx = dd
        elif args.depth_root:
            depth_root = Path(args.depth_root).expanduser().resolve()
            if depth_root.exists():
                depth_map = index_depth_npz(depth_root)
                video_root = video_path.parent
                npz_path = resolve_depth_npz_for_video(depth_root, video_root, video_path, depth_map)
                if npz_path and npz_path.exists():
                    dd = load_depth_npz_once(npz_path)
                    if dd:
                        depth_stack, depth_idx = dd
        if depth_stack is None:
            print("[WARN] No depth .npz found. Using zeros for depth input.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frames_per_clip = getattr(model, "frames_per_clip", 16)
    crop_size = getattr(model, "crop_size", None)
    if crop_size is None:
        if hasattr(model, "vjepa2"):
            crop_size = getattr(model.vjepa2.config, "crop_size", 256)
        else:
            crop_size = 256

    hist_frames = int(round(args.hist_s * fps))
    hist_frames = max(hist_frames, 1)
    stride_frames = max(1, int(round(args.stride_s * fps)))

    buffer_len = max(hist_frames, frames_per_clip)
    buffer_rgb = deque(maxlen=buffer_len)
    buffer_idx = deque(maxlen=buffer_len)

    last_pred_frame = -stride_frames
    last_prob = None
    times: List[float] = []
    probs: List[float] = []

    frame_idx = -1
    start_wall = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_sec = frame_idx / fps

        if args.max_seconds is not None and t_sec > args.max_seconds:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        buffer_rgb.append(frame_rgb)
        buffer_idx.append(frame_idx)

        can_infer = (frame_idx - last_pred_frame) >= stride_frames and len(buffer_rgb) >= max(frames_per_clip, hist_frames)
        if can_infer:
            last_pred_frame = frame_idx
            # sample indices across buffer
            n_buf = len(buffer_rgb)
            if n_buf == 1:
                sel = np.zeros((frames_per_clip,), dtype=np.int64)
            else:
                sel = np.linspace(0, n_buf - 1, frames_per_clip).astype(np.int64)

            clip_rgb = np.stack([buffer_rgb[i] for i in sel], axis=0)  # [T,H,W,C]
            inputs = processor([clip_rgb], return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad():
                if model_type == "rgb":
                    logits = model(**inputs)
                else:
                    if depth_stack is not None and depth_idx is not None:
                        idx_sel = np.array([buffer_idx[i] for i in sel], dtype=np.int64)
                        depth_clip = align_depth_nearest_fast(depth_stack, depth_idx, idx_sel)
                        depth_clip = resize_depth_clip(depth_clip, int(crop_size))
                    else:
                        depth_clip = np.zeros((frames_per_clip, int(crop_size), int(crop_size)), dtype=np.float32)
                    depth_t = preprocess_depth_to_tensor(depth_clip, device)
                    logits = model(pixel_values_videos=inputs["pixel_values_videos"], depth_videos=depth_t)

            prob = torch.sigmoid(logits).detach().float().cpu().numpy().flatten()[0]
            last_prob = float(prob)
            times.append(t_sec)
            probs.append(last_prob)
            print(f"[PRED] t={t_sec:6.2f}s prob={last_prob:0.4f}")

        if not args.no_display:
            disp = draw_overlay(frame_bgr, last_prob, t_sec, t_event)
            cv2.imshow("demo_video", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if args.save_csv:
        out_path = Path(args.save_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "prob"])
            for t, p in zip(times, probs):
                w.writerow([f"{t:.3f}", f"{p:.6f}"])
        print(f"[INFO] Saved CSV: {out_path}")

    elapsed = time.time() - start_wall
    print(f"[DONE] Processed {frame_idx + 1} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
