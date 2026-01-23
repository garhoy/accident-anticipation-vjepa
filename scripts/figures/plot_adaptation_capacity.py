#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG2: Effect of adaptation capacity (AP/AUC vs #unfrozen blocks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEAD_COLS = ("head", "head_type", "model", "name")
UNFREEZE_COLS = ("unfreeze", "unfrozen", "unfreeze_blocks", "blocks")
AP_COLS = ("ap", "AP")
AUC_COLS = ("auc", "AUC")
MTTA_COLS = ("mtta", "mtta_s", "tta", "mTTA")


def _load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --config.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    hcol = _find_col(df.columns, HEAD_COLS)
    ucol = _find_col(df.columns, UNFREEZE_COLS)
    apcol = _find_col(df.columns, AP_COLS)
    auccol = _find_col(df.columns, AUC_COLS)
    mttacol = _find_col(df.columns, MTTA_COLS)
    if ucol is None or apcol is None or auccol is None:
        raise ValueError("CSV needs columns for unfreeze, ap, auc.")
    if hcol is None:
        df["head"] = "default"
        hcol = "head"
    out = pd.DataFrame(
        {
            "head": df[hcol],
            "unfreeze": df[ucol],
            "ap": df[apcol],
            "auc": df[auccol],
        }
    )
    if mttacol is not None:
        out["mtta"] = df[mttacol]
    return out


def _load_metrics_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if isinstance(data, dict):
        rows = []
        for head, block_map in data.items():
            for unfreeze, vals in block_map.items():
                row = {"head": head, "unfreeze": unfreeze}
                row.update(vals)
                rows.append(row)
        data = rows
    if not isinstance(data, list):
        raise ValueError("JSON must be list of rows or dict-of-dicts.")
    df = pd.DataFrame(data)
    if "head" not in df.columns:
        df["head"] = "default"
    if "unfreeze" not in df.columns:
        raise ValueError("JSON missing 'unfreeze' field.")
    if "ap" not in df.columns or "auc" not in df.columns:
        raise ValueError("JSON missing 'ap' or 'auc'.")
    return df[["head", "unfreeze", "ap", "auc"] + (["mtta"] if "mtta" in df.columns else [])]


def _load_metrics(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return _load_metrics_csv(path)
    if path.suffix.lower() == ".json":
        return _load_metrics_json(path)
    raise ValueError("metrics file must be .csv or .json")


def _plot_head(ax, sub, x_order, mtta_mode):
    ap = sub.set_index("unfreeze").reindex(x_order)["ap"].to_numpy(dtype=float)
    auc = sub.set_index("unfreeze").reindex(x_order)["auc"].to_numpy(dtype=float)
    ax.plot(x_order, ap, marker="o", color="#1f77b4", label="AP")
    ax.plot(x_order, auc, marker="s", color="#ff7f0e", label="AUC")
    ax.set_xlabel("# unfrozen blocks")
    ax.set_ylabel("AP / AUC")
    ax.set_xticks(x_order)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    if "mtta" in sub.columns and mtta_mode != "none":
        mtta = sub.set_index("unfreeze").reindex(x_order)["mtta"].to_numpy(dtype=float)
        if mtta_mode.startswith("secondary"):
            ax2 = ax.twinx()
            if mtta_mode.endswith("bar"):
                ax2.bar(np.array(x_order) + 0.15, mtta, width=0.3, color="#2ca02c", alpha=0.35, label="mTTA")
            else:
                ax2.plot(x_order, mtta, marker="^", linestyle="--", color="#2ca02c", label="mTTA")
            ax2.set_ylabel("mTTA@0.50 (s)")
            ax.plot([], [], marker="^", linestyle="--", color="#2ca02c", label="mTTA")
        else:
            if mtta_mode == "bar":
                ax.bar(np.array(x_order) + 0.15, mtta, width=0.3, color="#2ca02c", alpha=0.35, label="mTTA")
            else:
                ax.plot(x_order, mtta, marker="^", linestyle="--", color="#2ca02c", label="mTTA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--metrics-csv", type=str, default=None)
    ap.add_argument("--metrics-json", type=str, default=None)
    ap.add_argument("--heads", type=str, nargs="*", default=None)
    ap.add_argument("--layout", type=str, default="subplots", choices=["subplots", "single"])
    ap.add_argument(
        "--mtta-mode",
        type=str,
        default="none",
        choices=["none", "line", "bar", "secondary-line", "secondary-bar"],
    )
    ap.add_argument("--x-order", type=int, nargs="*", default=[0, 2, 4, 24])
    ap.add_argument("--out", type=str, default="Imagenes/adaptation_capacity_rgb.png")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    metrics_path = args.metrics_csv or args.metrics_json or cfg.get("metrics_path")
    if metrics_path is None:
        raise ValueError("Provide --metrics-csv or --metrics-json.")
    metrics_path = Path(metrics_path)

    df = _load_metrics(metrics_path)
    df["unfreeze"] = df["unfreeze"].astype(int)
    heads = args.heads or cfg.get("heads") or sorted(df["head"].unique())
    x_order = args.x_order or cfg.get("x_order") or [0, 2, 4, 24]

    if args.layout == "single":
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        for head in heads:
            sub = df[df["head"] == head]
            ap_vals = sub.set_index("unfreeze").reindex(x_order)["ap"].to_numpy(dtype=float)
            auc_vals = sub.set_index("unfreeze").reindex(x_order)["auc"].to_numpy(dtype=float)
            ax.plot(x_order, ap_vals, marker="o", label=f"{head}-AP")
            ax.plot(x_order, auc_vals, marker="s", label=f"{head}-AUC")
        ax.set_xlabel("# unfrozen blocks")
        ax.set_ylabel("AP / AUC")
        ax.set_xticks(x_order)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
    else:
        n = len(heads)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, head in zip(axes, heads):
            sub = df[df["head"] == head]
            _plot_head(ax, sub, x_order, args.mtta_mode)
            ax.set_title(head)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
