#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG3: Case study - 5 frames + prob curve.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils_io import load_curve, prob_at_time


def _load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --config.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_frame_at_time(cap: cv2.VideoCapture, t_sec: float, fps: float) -> Optional[np.ndarray]:
    if fps > 0:
        frame_idx = int(round(t_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def _plot_case(
    frames: List[np.ndarray],
    times_sel: List[float],
    probs_sel: List[float],
    times_curve: np.ndarray,
    probs_curve: np.ndarray,
    t_event: float,
    theta: float,
    out_path: Path,
) -> None:
    n = len(frames)
    fig = plt.figure(figsize=(2.6 * n, 6.0))
    gs = fig.add_gridspec(2, n, height_ratios=[2.2, 1.2], hspace=0.15)

    for i, (frame, t, p) in enumerate(zip(frames, times_sel, probs_sel)):
        ax = fig.add_subplot(gs[0, i])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(f"t={t:.2f}s\np={p:.2f}")

    axc = fig.add_subplot(gs[1, :])
    axc.plot(times_curve, probs_curve, color="#d62728", linewidth=2.0, label="p_w")
    axc.fill_between(times_curve, probs_curve, color="#d62728", alpha=0.12)
    axc.axvline(x=t_event, color="#2ca02c", linestyle="--", linewidth=2, label="t_event")
    axc.axhline(y=theta, color="gray", linestyle=":", linewidth=1.0, label="theta")
    axc.scatter(times_sel, probs_sel, color="#1f77b4", s=40, zorder=5, label="selected")
    axc.set_xlabel("decision time t_dec (s)")
    axc.set_ylabel("accident prob")
    axc.set_ylim(0, 1.05)
    axc.grid(True, alpha=0.3)
    axc.legend(loc="upper left", ncol=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--curve", type=str, default=None)
    ap.add_argument("--t-event", type=float, default=None)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--times", type=float, nargs="+", default=None)
    ap.add_argument("--interp", type=str, default="linear", choices=["linear", "nearest"])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    video = args.video or cfg.get("video")
    curve = args.curve or cfg.get("curve")
    t_event = args.t_event if args.t_event is not None else cfg.get("t_event")
    times_sel = args.times or cfg.get("times")
    theta = args.theta if args.theta is not None else cfg.get("theta", 0.5)
    out = args.out or cfg.get("out")

    if video is None or curve is None or t_event is None or times_sel is None:
        raise ValueError("Provide --video, --curve, --t-event, and --times (or config).")

    video_path = Path(video).expanduser().resolve()
    curve_path = Path(curve).expanduser().resolve()
    if out is None:
        clip_id = video_path.stem
        out = f"Imagenes/casestudy_frames_prob_{clip_id}.png"
    out_path = Path(out).expanduser().resolve()

    times_curve, probs_curve = load_curve(curve_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frames: List[np.ndarray] = []
    probs_sel: List[float] = []
    for t in times_sel:
        frame = _get_frame_at_time(cap, float(t), float(fps))
        if frame is None:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 224)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 224)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        frames.append(frame)
        probs_sel.append(prob_at_time(times_curve, probs_curve, float(t), method=args.interp))

    cap.release()

    _plot_case(frames, [float(t) for t in times_sel], probs_sel, times_curve, probs_curve, float(t_event), float(theta), out_path)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
