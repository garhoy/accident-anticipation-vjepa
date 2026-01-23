#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_video_event_tail.py

Genera un video lento + grafica de probabilidad para los ultimos segundos
antes del accidente (usa time_of_event del CSV o un valor manual).

Ejemplo:
  .venv/bin/python scripts/demos/demo_video_event_tail.py \
    --video 00037.mp4 \
    --csv data/metadata/Nexar/train.csv \
    --checkpoint checkpoints/depth_vjepa/checkpoints_vjepa_tf_rgb_frozen_hn05_j0-2_pw3/best_robust_model.pt \
    --model-type auto \
    --head-type transformer \
    --tail-s 5 \
    --slow-factor 4
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoVideoProcessor

# Reuse utilities/models from demo_video.py
from demo_video import (
    VJEPA2TemporalBinary,
    _import_depth_models,
    index_depth_npz,
    resolve_depth_npz_for_video,
    load_depth_npz_once,
    align_depth_nearest_fast,
    resize_depth_clip,
    preprocess_depth_to_tensor,
    infer_model_type,
    pick_arg,
    lookup_time_of_event,
    draw_overlay,
    VJEPA2DepthTransformerBinary_DepthNet,
    VJEPA2DepthTransformerBinary_Patchwise,
    VJEPA2DepthMidFiLMBinary,
)


def _load_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("args", {}) or {}
    return ckpt, {}


def _build_model(
    state: Dict[str, torch.Tensor],
    ckpt_args: Dict,
    model_type: str,
    hf_repo: str,
    head_type: str,
    depth_dim: int,
    midfusion_depth_dim: Optional[int],
) -> torch.nn.Module:
    if model_type == "rgb":
        return VJEPA2TemporalBinary(hf_repo, head_type=head_type)

    _import_depth_models()
    spatial_heads = pick_arg(ckpt_args, "spatial_heads", 1)
    encoder_ckpt = pick_arg(ckpt_args, "encoder_ckpt", None)

    if model_type == "depth_net":
        return VJEPA2DepthTransformerBinary_DepthNet(  # type: ignore
            hf_repo, depth_dim=depth_dim, unfreeze_blocks=0, encoder_ckpt=encoder_ckpt
        )
    if model_type == "patchwise":
        return VJEPA2DepthTransformerBinary_Patchwise(  # type: ignore
            hf_repo, depth_dim=depth_dim, spatial_heads=spatial_heads, unfreeze_blocks=0, encoder_ckpt=encoder_ckpt
        )
    if model_type == "midfusion":
        if VJEPA2DepthMidFiLMBinary is None:
            raise RuntimeError("Mid-fusion model not available in this repo/env.")
        mf_depth_dim = midfusion_depth_dim if midfusion_depth_dim is not None else depth_dim
        inject_layers = pick_arg(ckpt_args, "inject_layers", "-4,-3,-2,-1")
        film_hidden = pick_arg(ckpt_args, "film_hidden", 256)
        film_scale = pick_arg(ckpt_args, "film_scale", 0.10)
        per_layer_adapters = pick_arg(ckpt_args, "per_layer_adapters", True)
        return VJEPA2DepthMidFiLMBinary(  # type: ignore
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
    raise ValueError(f"Unknown model type: {model_type}")


def _save_plot(times: List[float], probs: List[float], t_event: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(times, probs, color="#d62728", linewidth=2.0, label="Prob")
    plt.fill_between(times, probs, color="#d62728", alpha=0.12)
    plt.axvline(x=t_event, color="#2ca02c", linestyle="--", linewidth=2, label=f"Event t={t_event:.2f}s")
    plt.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.0, label="thr=0.5")
    plt.ylim(0, 1.05)
    plt.xlabel("Time (s)")
    plt.ylabel("Accident prob")
    plt.title("Probability over last seconds before event")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--time-of-event", type=float, default=None)
    ap.add_argument("--model-type", type=str, default="auto", choices=["auto", "rgb", "depth_net", "patchwise", "midfusion"])
    ap.add_argument("--head-type", type=str, default="transformer")
    ap.add_argument("--hf-repo", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tail-s", type=float, default=5.0, help="Seconds before event to analyze.")
    ap.add_argument("--hist-s", type=float, default=5.0, help="History seconds per clip.")
    ap.add_argument("--stride-frames", type=int, default=1, help="Stride in frames for predictions.")
    ap.add_argument("--slow-factor", type=float, default=4.0, help="Output video is fps/slow_factor.")
    ap.add_argument("--out-video", type=str, default=None)
    ap.add_argument("--out-plot", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--depth-root", type=str, default=None)
    ap.add_argument("--depth-npz", type=str, default=None)
    ap.add_argument("--depth-dim", type=int, default=128)
    ap.add_argument("--midfusion-depth-dim", type=int, default=None)
    args = ap.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)

    state, ckpt_args = _load_checkpoint(ckpt_path)
    model_type = infer_model_type(state, args.model_type)
    hf_repo = pick_arg(ckpt_args, "hf_repo", args.hf_repo)
    head_type = pick_arg(ckpt_args, "head_type", args.head_type)
    depth_dim = pick_arg(ckpt_args, "depth_dim", args.depth_dim)

    model = _build_model(state, ckpt_args, model_type, hf_repo, head_type, depth_dim, args.midfusion_depth_dim)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    # time_of_event
    t_event = args.time_of_event
    if t_event is None:
        t_event = lookup_time_of_event(Path(args.csv) if args.csv else None, video_path.stem)
    if t_event is None:
        raise ValueError("time_of_event not found. Provide --time-of-event or --csv with time_of_event.")

    # depth (optional)
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

    hist_frames = max(1, int(round(args.hist_s * fps)))
    stride_frames = max(1, int(args.stride_frames))
    tail_frames = max(1, int(round(args.tail_s * fps)))

    warmup_s = max(0.0, float(t_event) - float(args.tail_s) - float(args.hist_s))
    start_s = max(0.0, float(t_event) - float(args.tail_s))
    end_s = float(t_event)

    start_frame = int(round(warmup_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # outputs
    out_video = args.out_video
    out_plot = args.out_plot
    out_csv = args.out_csv
    if out_video is None:
        out_video = f"results/tail_{video_path.stem}_slow.mp4"
    if out_plot is None:
        out_plot = f"results/tail_{video_path.stem}_plot.png"

    out_video_path = Path(out_video).expanduser().resolve()
    out_plot_path = Path(out_plot).expanduser().resolve()

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)

    # writer
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    h, w = first_frame.shape[:2]
    slow_fps = float(fps) / max(1e-6, float(args.slow_factor))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, slow_fps, (w, h))

    # rewind to start_frame again after peeking
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    buffer_rgb: List[np.ndarray] = []
    buffer_idx: List[int] = []
    max_buf = max(hist_frames, frames_per_clip)
    last_pred_frame = -stride_frames
    last_prob = None

    times: List[float] = []
    probs: List[float] = []

    frame_idx = start_frame - 1
    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            t_sec = frame_idx / fps

            if t_sec > end_s:
                break

            # buffer
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buffer_rgb.append(frame_rgb)
            buffer_idx.append(frame_idx)
            if len(buffer_rgb) > max_buf:
                buffer_rgb.pop(0)
                buffer_idx.pop(0)

            if t_sec < start_s:
                continue

            can_infer = (frame_idx - last_pred_frame) >= stride_frames and len(buffer_rgb) >= max_buf
            if can_infer:
                last_pred_frame = frame_idx
                n_buf = len(buffer_rgb)
                if n_buf == 1:
                    sel = np.zeros((frames_per_clip,), dtype=np.int64)
                else:
                    sel = np.linspace(0, n_buf - 1, frames_per_clip).astype(np.int64)

                clip_rgb = np.stack([buffer_rgb[i] for i in sel], axis=0)
                inputs = processor([clip_rgb], return_tensors="pt")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

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

            # write frame (slow)
            disp = draw_overlay(frame_bgr, last_prob, t_sec, t_event)
            writer.write(disp)

    cap.release()
    writer.release()

    _save_plot(times, probs, t_event, out_plot_path)

    if out_csv:
        out_csv_path = Path(out_csv).expanduser().resolve()
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with out_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "prob"])
            for t, p in zip(times, probs):
                w.writerow([f"{t:.3f}", f"{p:.6f}"])
        print(f"[INFO] Saved CSV: {out_csv_path}")

    print(f"[DONE] Video: {out_video_path}")
    print(f"[DONE] Plot:  {out_plot_path}")


if __name__ == "__main__":
    main()
