#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG1: Qualitative risk curves (RGB-only).

Creates a 1x2 figure with two curves:
  (a) early but noisy detection
  (b) false alarm (negative)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from utils_io import first_crossing_time, load_curve


def _load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --config.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick(val, cfg, key, default=None):
    if val is not None:
        return val
    if key in cfg:
        return cfg[key]
    return default


def _annotate_pos(ax, t_alarm, t_event):
    if t_alarm is None:
        ax.text(0.02, 0.92, "miss", transform=ax.transAxes, fontsize=10, color="black")
        return
    ax.axvline(x=t_alarm, color="#ff7f0e", linestyle="--", linewidth=1.5, label="t_alarm")
    if t_alarm <= t_event:
        tta = t_event - t_alarm
        msg = f"t_alarm={t_alarm:.2f}s\nTTA={tta:.2f}s"
    else:
        msg = f"t_alarm={t_alarm:.2f}s\n(after event)"
    ax.text(0.02, 0.92, msg, transform=ax.transAxes, fontsize=9, color="black", va="top")


def _annotate_neg(ax, t_alarm):
    if t_alarm is not None:
        ax.axvline(x=t_alarm, color="#ff7f0e", linestyle="--", linewidth=1.5, label="t_alarm")
        msg = f"t_alarm={t_alarm:.2f}s"
    else:
        msg = "no alarm"
    ax.text(0.02, 0.92, msg, transform=ax.transAxes, fontsize=9, color="black", va="top")


def plot_subplot(ax, times, probs, t_event, theta, title, is_pos):
    ax.plot(times, probs, color="#d62728", linewidth=2.0, label="p_w")
    ax.fill_between(times, probs, color="#d62728", alpha=0.12)
    ax.axhline(y=theta, color="gray", linestyle=":", linewidth=1.0, label="theta")
    t_alarm = first_crossing_time(times, probs, theta)
    if is_pos:
        ax.axvline(x=t_event, color="#2ca02c", linestyle="--", linewidth=2, label="t_event")
        _annotate_pos(ax, t_alarm, t_event)
    else:
        _annotate_neg(ax, t_alarm)
    ax.set_title(title)
    ax.set_xlabel("decision time t_dec (s)")
    ax.set_ylabel("accident prob")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--input-a", type=str, default=None)
    ap.add_argument("--input-b", type=str, default=None)
    ap.add_argument("--t-event", type=float, default=None)
    ap.add_argument("--t-event-a", type=float, default=None)
    ap.add_argument("--t-event-b", type=float, default=None)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-a", type=float, default=None)
    ap.add_argument("--theta-b", type=float, default=None)
    ap.add_argument("--label-a", type=str, default="pos", choices=["pos", "neg"])
    ap.add_argument("--label-b", type=str, default="neg", choices=["pos", "neg"])
    ap.add_argument("--title-a", type=str, default="(a) early but noisy")
    ap.add_argument("--title-b", type=str, default="(b) false alarm (negative)")
    ap.add_argument("--modeltag", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    input_a = _pick(args.input_a, cfg, "input_a")
    input_b = _pick(args.input_b, cfg, "input_b")
    if input_a is None or input_b is None:
        raise ValueError("Provide --input-a and --input-b (or config).")

    t_event = _pick(args.t_event, cfg, "t_event")
    t_event_a = _pick(args.t_event_a, cfg, "t_event_a", t_event)
    t_event_b = _pick(args.t_event_b, cfg, "t_event_b", t_event)
    if args.label_a == "pos" and t_event_a is None:
        raise ValueError("Positive panel requires --t-event-a (or --t-event).")
    if args.label_b == "pos" and t_event_b is None:
        raise ValueError("Positive panel requires --t-event-b (or --t-event).")

    theta = _pick(args.theta, cfg, "theta", 0.5)
    theta_a = _pick(args.theta_a, cfg, "theta_a", theta)
    theta_b = _pick(args.theta_b, cfg, "theta_b", theta)

    panel_a = _pick(args.label_a, cfg, "label_a", "pos")
    panel_b = _pick(args.label_b, cfg, "label_b", "neg")
    title_a = _pick(args.title_a, cfg, "title_a", "(a) early but noisy")
    title_b = _pick(args.title_b, cfg, "title_b", "(b) false alarm (negative)")

    if args.out:
        out_path = Path(args.out)
    else:
        modeltag = _pick(args.modeltag, cfg, "modeltag", "rgb")
        out_path = Path(f"Imagenes/rgb_risk_curves_{modeltag}.png")

    times_a, probs_a = load_curve(input_a)
    times_b, probs_b = load_curve(input_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_subplot(axes[0], times_a, probs_a, float(t_event_a or 0.0), float(theta_a), title_a, panel_a == "pos")
    plot_subplot(axes[1], times_b, probs_b, float(t_event_b or 0.0), float(theta_b), title_b, panel_b == "pos")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
