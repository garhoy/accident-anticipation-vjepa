#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG: RGBD qualitative case study - 5 RGB frames + 5 depth frames + risk curves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils_io import (  # noqa: E402
    extract_frame,
    first_crossing_time,
    index_frames_dir,
    load_curve,
    open_video,
    prob_at_time,
    read_image,
)


def _parse_t_event(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "nan", ""}:
        return None
    return float(value)


def _format_timestamp_title(t: float) -> str:
    return f"t = {t:.1f} s"


def _build_prob_table_rows(
    times_sel: List[float],
    probs_rgb_sel: List[float],
    probs_rgbd_sel: List[float],
) -> List[List[str]]:
    rows: List[List[str]] = []
    for t, p_rgb, p_rgbd in zip(times_sel, probs_rgb_sel, probs_rgbd_sel):
        rows.append([f"{t:.1f}", f"{p_rgb:.2f}", f"{p_rgbd:.2f}"])
    return rows


def _compute_xlim(
    mode: str,
    times_rgb: np.ndarray,
    times_rgbd: np.ndarray,
    times_sel: List[float],
    t_event: Optional[float],
    t_alarm_rgb: Optional[float],
    t_alarm_rgbd: Optional[float],
) -> Tuple[float, float]:
    if mode == "zoom":
        alarm_times = [t for t in (t_alarm_rgb, t_alarm_rgbd) if t is not None]
        if t_event is not None:
            if alarm_times:
                xmin = min(alarm_times) - 2.0
            else:
                xmin = min(times_sel) - 1.0
            xmax = t_event + 1.0
        else:
            if alarm_times:
                alarm_time = min(alarm_times)
                xmin = alarm_time - 2.0
                xmax = alarm_time + 6.0
            else:
                xmin = min(times_sel) - 1.0
                xmax = max(times_sel) + 1.0
        if xmin >= xmax:
            xmin = min(times_sel) - 1.0
            xmax = max(times_sel) + 1.0
        return float(xmin), float(xmax)

    x_candidates = list(times_rgb) + list(times_rgbd) + list(times_sel)
    if t_event is not None:
        x_candidates.append(float(t_event))
    if t_alarm_rgb is not None:
        x_candidates.append(float(t_alarm_rgb))
    if t_alarm_rgbd is not None:
        x_candidates.append(float(t_alarm_rgbd))
    xmin = min(x_candidates)
    xmax = max(x_candidates)
    pad = 0.02 * (xmax - xmin) if xmax > xmin else 1.0
    return float(xmin - pad), float(xmax + pad)


def _add_prob_table(
    ax: plt.Axes,
    times_sel: List[float],
    probs_rgb_sel: List[float],
    probs_rgbd_sel: List[float],
    font_size: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    col_labels = ["t", "p_rgb", "p_rgbd"]
    cell_text = _build_prob_table_rows(times_sel, probs_rgb_sel, probs_rgbd_sel)
    if bbox is None:
        bbox = (0.0, -0.36, 1.0, 0.22)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.set_clip_on(False)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("0.7")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("0.95")
            cell.set_text_props(weight="bold")
    table.scale(1.0, 1.0)


def _shade_above_threshold(
    ax: plt.Axes,
    times: np.ndarray,
    probs: np.ndarray,
    theta: float,
    color: str,
    xlim: Tuple[float, float],
    alpha: float,
) -> None:
    times = np.asarray(times)
    probs = np.asarray(probs)
    mask = (times >= xlim[0]) & (times <= xlim[1])
    if not np.any(mask):
        return
    times_sel = times[mask]
    probs_sel = probs[mask]
    ax.fill_between(
        times_sel,
        probs_sel,
        theta,
        where=probs_sel >= theta,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        zorder=1,
    )


def _label_vertical_line(
    ax: plt.Axes,
    x: Optional[float],
    text: str,
    color: str,
    font_size: float,
    alpha: float = 1.0,
) -> None:
    if x is None:
        return
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.02 * (ymax - ymin)
    ax.text(
        x,
        y_text,
        text,
        rotation=90,
        ha="right",
        va="top",
        fontsize=max(font_size - 1.0, 7.0),
        color=color,
        alpha=alpha,
    )


def _count_axes_overlaps(fig: plt.Figure) -> Tuple[int, List[Tuple[str, str]]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes: List[Tuple[str, matplotlib.transforms.Bbox]] = []
    for idx, ax in enumerate(fig.axes):
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        label = ax.get_label() or f"axes_{idx}"
        bboxes.append((label, bbox))
    overlaps: List[Tuple[str, str]] = []
    for i, (label_i, bbox_i) in enumerate(bboxes):
        for label_j, bbox_j in bboxes[i + 1 :]:
            if bbox_i.overlaps(bbox_j):
                overlaps.append((label_i, label_j))
    return len(overlaps), overlaps


def _prepare_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.stack([frame] * 3, axis=-1)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _prepare_depth(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _load_video_frames(video_path: Path, times: List[float], label: str) -> List[np.ndarray]:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = None
    try:
        cap = open_video(video_path)
    except Exception as exc:
        print(
            f"[WARN] OpenCV could not open {label} video ({video_path}); falling back to ffmpeg. {exc}",
            file=sys.stderr,
        )
    frames: List[np.ndarray] = []
    for t in times:
        frame = extract_frame(video_path, t, cap=cap)
        if frame is None:
            if cap is not None:
                cap.release()
            raise RuntimeError(f"Failed to read {label} frame at t={t:.3f}s from {video_path}")
        frames.append(frame)
    if cap is not None:
        cap.release()
    return frames


def _load_depth_frames_from_dir(
    frames_dir: Path,
    times: List[float],
    fps: Optional[float],
) -> List[np.ndarray]:
    times_all, paths_all = index_frames_dir(frames_dir, fps=fps)
    frames: List[np.ndarray] = []
    for t in times:
        idx = int(np.argmin(np.abs(times_all - float(t))))
        frame = read_image(paths_all[idx])
        frames.append(frame)
    return frames


def _export_debug(
    out_dir: Path,
    times: List[float],
    rgb_frames: List[np.ndarray],
    depth_frames: List[np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (t, rgb, depth) in enumerate(zip(times, rgb_frames, depth_frames), start=1):
        rgb_path = out_dir / f"rgb_t{i}_{t:.3f}s.png"
        depth_path = out_dir / f"depth_t{i}_{t:.3f}s.png"
        cv2.imwrite(str(rgb_path), rgb)
        cv2.imwrite(str(depth_path), depth)


def _plot_figure(
    rgb_frames: List[np.ndarray],
    depth_frames: List[np.ndarray],
    times_sel: List[float],
    probs_rgb_sel: List[float],
    probs_rgbd_sel: List[float],
    times_rgb: np.ndarray,
    probs_rgb: np.ndarray,
    times_rgbd: np.ndarray,
    probs_rgbd: np.ndarray,
    t_event: Optional[float],
    theta: float,
    t_alarm_rgb: Optional[float],
    t_alarm_rgbd: Optional[float],
    out_path: Path,
    save_pdf: bool,
    xlim_mode: str,
    font_size: float,
    title_font_size: float,
    shading: bool,
) -> Tuple[int, List[Tuple[str, str]]]:
    n = len(times_sel)
    fig = plt.figure(figsize=(2.8 * n, 6.4))
    with plt.rc_context(
        {
            "font.size": font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    ):
        gs = fig.add_gridspec(
            3,
            n,
            height_ratios=[1.05, 1.05, 1.25],
            left=0.03,
            right=0.995,
            top=0.97,
            bottom=0.11,
            hspace=0.08,
            wspace=0.02,
        )

        for i in range(n):
            t = float(times_sel[i])
            title = _format_timestamp_title(t)

            ax_rgb = fig.add_subplot(gs[0, i])
            ax_rgb.set_label(f"rgb_{i + 1}")
            ax_rgb.imshow(_prepare_rgb(rgb_frames[i]))
            ax_rgb.set_title(title, pad=2.0)
            ax_rgb.axis("off")

            ax_depth = fig.add_subplot(gs[1, i])
            ax_depth.set_label(f"depth_{i + 1}")
            ax_depth.imshow(_prepare_depth(depth_frames[i]), cmap="gray")
            ax_depth.set_title(title, pad=2.0)
            ax_depth.axis("off")

        ax = fig.add_subplot(gs[2, :])
        ax.set_label("risk_plot")

        color_rgb = "#1f77b4"
        color_rgbd = "#d62728"

        line_rgb = ax.plot(
            times_rgb,
            probs_rgb,
            color=color_rgb,
            linewidth=1.8,
            linestyle="-",
            label="RGB-only",
            zorder=3,
        )[0]
        line_rgbd = ax.plot(
            times_rgbd,
            probs_rgbd,
            color=color_rgbd,
            linewidth=1.8,
            linestyle="-",
            label="RGB+Depth",
            zorder=3,
        )[0]

        ax.plot(
            times_sel,
            probs_rgb_sel,
            linestyle="None",
            marker="o",
            markersize=3.5,
            color=color_rgb,
            zorder=4,
        )
        ax.plot(
            times_sel,
            probs_rgbd_sel,
            linestyle="None",
            marker="o",
            markersize=3.5,
            color=color_rgbd,
            zorder=4,
        )

        theta_label = f"θ={theta:.2f}"
        theta_line = ax.axhline(
            y=theta,
            color="0.4",
            linestyle=":",
            linewidth=1.0,
            label=theta_label,
            zorder=2,
        )

        if t_event is not None:
            ax.axvline(x=t_event, color="#2ca02c", linestyle="--", linewidth=1.2)
        if t_alarm_rgb is not None:
            ax.axvline(
                x=t_alarm_rgb,
                color=color_rgb,
                linestyle="--",
                linewidth=0.9,
                alpha=0.6,
            )
        if t_alarm_rgbd is not None:
            ax.axvline(
                x=t_alarm_rgbd,
                color=color_rgbd,
                linestyle="--",
                linewidth=0.9,
                alpha=0.6,
            )

        xlim = _compute_xlim(
            xlim_mode,
            times_rgb=times_rgb,
            times_rgbd=times_rgbd,
            times_sel=times_sel,
            t_event=t_event,
            t_alarm_rgb=t_alarm_rgb,
            t_alarm_rgbd=t_alarm_rgbd,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("time (s)", labelpad=1.0)
        ax.set_ylabel("risk")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if shading and xlim_mode == "zoom":
            _shade_above_threshold(ax, times_rgb, probs_rgb, theta, color_rgb, xlim, alpha=0.12)
            _shade_above_threshold(
                ax, times_rgbd, probs_rgbd, theta, color_rgbd, xlim, alpha=0.12
            )

        _label_vertical_line(ax, t_event, "event", "#2ca02c", font_size)
        _label_vertical_line(ax, t_alarm_rgb, "alarm rgb", color_rgb, font_size, alpha=0.6)
        _label_vertical_line(ax, t_alarm_rgbd, "alarm rgbd", color_rgbd, font_size, alpha=0.6)

        ax.legend(handles=[line_rgb, line_rgbd, theta_line], loc="upper left", ncol=3, frameon=False)

        _add_prob_table(ax, times_sel, probs_rgb_sel, probs_rgbd_sel, font_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_count, overlap_pairs = _count_axes_overlaps(fig)
    fig.savefig(out_path, dpi=300)
    if save_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path)
    plt.close(fig)
    return overlap_count, overlap_pairs


def _latex_block(out_path: Path) -> str:
    caption = (
        "Qualitative RGB-D case study with five RGB frames (top), aligned depth maps "
        "(middle), and streaming risk probabilities (bottom) for RGB-only and RGB+Depth. "
        "Vertical lines mark event/alarm times and the horizontal line is the threshold."
    )
    return (
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{\\detokenize{{{out_path}}}}}\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{fig:rgbd_case_study}\n"
        "\\end{figure}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-video", type=str, required=True)
    ap.add_argument("--depth-video", type=str, default=None)
    ap.add_argument("--depth-frames-dir", type=str, default=None)
    ap.add_argument("--depth-fps", type=float, default=None)
    ap.add_argument("--curve-rgb", type=str, required=True)
    ap.add_argument("--curve-rgbd", type=str, required=True)
    ap.add_argument("--times", type=float, nargs=5, required=True)
    ap.add_argument("--t-event", type=str, default="none")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--interp", type=str, default="nearest", choices=["nearest", "linear"])
    ap.add_argument("--out", type=str, default="Imagenes/Fig_RGB_Depth_Qualitative.png")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--export-debug", type=str, default=None)
    ap.add_argument("--xlim-mode", type=str, choices=["full", "zoom"], default=None)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--font-size", type=float, default=9.0)
    ap.add_argument("--title-font-size", type=float, default=9.0)
    ap.add_argument("--no-shading", action="store_true")
    args = ap.parse_args()

    if args.depth_video and args.depth_frames_dir:
        raise ValueError("Provide either --depth-video or --depth-frames-dir, not both.")
    if not args.depth_video and not args.depth_frames_dir:
        raise ValueError("Provide --depth-video or --depth-frames-dir.")

    rgb_video = Path(args.rgb_video).expanduser().resolve()
    curve_rgb_path = Path(args.curve_rgb).expanduser().resolve()
    curve_rgbd_path = Path(args.curve_rgbd).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    times_sel = [float(t) for t in args.times]
    if len(times_sel) != 5:
        raise ValueError("Expected exactly 5 times for --times.")

    t_event = _parse_t_event(args.t_event)

    times_rgb, probs_rgb = load_curve(curve_rgb_path)
    times_rgbd, probs_rgbd = load_curve(curve_rgbd_path)

    probs_rgb_sel = [
        prob_at_time(times_rgb, probs_rgb, t, method=args.interp) for t in times_sel
    ]
    probs_rgbd_sel = [
        prob_at_time(times_rgbd, probs_rgbd, t, method=args.interp) for t in times_sel
    ]

    t_alarm_rgb = first_crossing_time(times_rgb, probs_rgb, float(args.theta))
    t_alarm_rgbd = first_crossing_time(times_rgbd, probs_rgbd, float(args.theta))

    rgb_frames = _load_video_frames(rgb_video, times_sel, label="RGB")
    if args.depth_video:
        depth_video = Path(args.depth_video).expanduser().resolve()
        depth_frames = _load_video_frames(depth_video, times_sel, label="depth")
    else:
        depth_frames_dir = Path(args.depth_frames_dir).expanduser().resolve()
        depth_frames = _load_depth_frames_from_dir(depth_frames_dir, times_sel, fps=args.depth_fps)

    if args.export_debug:
        debug_dir = Path(args.export_debug).expanduser().resolve()
        _export_debug(debug_dir, times_sel, rgb_frames, depth_frames)

    xlim_mode = args.xlim_mode
    if xlim_mode is None:
        xlim_mode = "zoom" if args.paper else "full"

    overlap_count, overlap_pairs = _plot_figure(
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        times_sel=times_sel,
        probs_rgb_sel=probs_rgb_sel,
        probs_rgbd_sel=probs_rgbd_sel,
        times_rgb=times_rgb,
        probs_rgb=probs_rgb,
        times_rgbd=times_rgbd,
        probs_rgbd=probs_rgbd,
        t_event=t_event,
        theta=float(args.theta),
        t_alarm_rgb=t_alarm_rgb,
        t_alarm_rgbd=t_alarm_rgbd,
        out_path=out_path,
        save_pdf=bool(args.pdf),
        xlim_mode=xlim_mode,
        font_size=float(args.font_size),
        title_font_size=float(args.title_font_size),
        shading=bool(args.paper) and not bool(args.no_shading),
    )

    print(f"[OK] Saved {out_path}")
    if args.pdf:
        print(f"[OK] Saved {out_path.with_suffix('.pdf')}")
    print(f"[OK] Layout overlap count: {overlap_count}")
    if overlap_pairs:
        print(f"[WARN] Layout overlaps: {overlap_pairs}")
    print("\nLaTeX:")
    print(_latex_block(out_path))


if __name__ == "__main__":
    main()
