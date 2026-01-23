#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O helpers for figure scripts (Nexar accident anticipation).

Supported curve formats:
  - CSV: columns include time_s, prob (case-insensitive, aliases allowed)
  - JSON: {"time_s":[...], "prob":[...]} or list of dicts
  - NPY: array Nx2 [time, prob] or 2xN

Video/image helpers:
  - robust frame extraction by timestamp (OpenCV, ffmpeg fallback)
  - depth frames directory indexing by timestamp or frame index
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_COLS = ("time_s", "time", "t", "t_sec", "t_s")
PROB_COLS = ("prob", "p", "score", "risk")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _ensure_sorted(times: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    return times[order], probs[order]


def load_curve(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    if ext == ".csv":
        return load_curve_csv(p)
    if ext == ".json":
        return load_curve_json(p)
    if ext == ".npy":
        return load_curve_npy(p)
    if ext == ".npz":
        return load_curve_npz(p)
    raise ValueError(f"Unsupported curve format: {p}")


def load_curve_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    tcol = _find_col(df.columns, TIME_COLS)
    pcol = _find_col(df.columns, PROB_COLS)
    if tcol is None or pcol is None:
        raise ValueError(f"CSV missing time/prob columns: {path}")
    times = df[tcol].to_numpy(dtype=float)
    probs = df[pcol].to_numpy(dtype=float)
    return _ensure_sorted(times, probs)


def load_curve_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "time_s" in data and "prob" in data:
            times = np.asarray(data["time_s"], dtype=float)
            probs = np.asarray(data["prob"], dtype=float)
            return _ensure_sorted(times, probs)
        if "times" in data and "probs" in data:
            times = np.asarray(data["times"], dtype=float)
            probs = np.asarray(data["probs"], dtype=float)
            return _ensure_sorted(times, probs)
    if isinstance(data, list):
        times = []
        probs = []
        for row in data:
            if isinstance(row, dict):
                t = row.get("time_s", row.get("time", row.get("t")))
                p = row.get("prob", row.get("p", row.get("score")))
                if t is None or p is None:
                    continue
                times.append(float(t))
                probs.append(float(p))
            else:
                if len(row) >= 2:
                    times.append(float(row[0]))
                    probs.append(float(row[1]))
        return _ensure_sorted(np.asarray(times), np.asarray(probs))
    raise ValueError(f"Unrecognized JSON format: {path}")


def load_curve_npy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}")
    if arr.shape[1] == 2:
        times = arr[:, 0].astype(float)
        probs = arr[:, 1].astype(float)
    elif arr.shape[0] == 2:
        times = arr[0].astype(float)
        probs = arr[1].astype(float)
    else:
        raise ValueError(f"Expected Nx2 or 2xN array in {path}")
    return _ensure_sorted(times, probs)


def load_curve_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "time_s" in data and "prob" in data:
        times = data["time_s"].astype(float)
        probs = data["prob"].astype(float)
        return _ensure_sorted(times, probs)
    if "times" in data and "probs" in data:
        times = data["times"].astype(float)
        probs = data["probs"].astype(float)
        return _ensure_sorted(times, probs)
    if "arr" in data:
        arr = data["arr"]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}")
        if arr.shape[1] == 2:
            times = arr[:, 0].astype(float)
            probs = arr[:, 1].astype(float)
        elif arr.shape[0] == 2:
            times = arr[0].astype(float)
            probs = arr[1].astype(float)
        else:
            raise ValueError(f"Expected Nx2 or 2xN array in {path}")
        return _ensure_sorted(times, probs)
    raise ValueError(f"Unrecognized NPZ format: {path}")


def prob_at_time(
    times: np.ndarray,
    probs: np.ndarray,
    t: float,
    method: str = "linear",
) -> float:
    if times.size == 0:
        return float("nan")
    if t <= times[0]:
        return float(probs[0])
    if t >= times[-1]:
        return float(probs[-1])
    if method == "nearest":
        idx = int(np.argmin(np.abs(times - t)))
        return float(probs[idx])
    # linear interp
    return float(np.interp(t, times, probs))


def first_crossing_time(times: np.ndarray, probs: np.ndarray, theta: float) -> Optional[float]:
    if times.size == 0:
        return None
    idx = np.where(probs >= theta)[0]
    if idx.size == 0:
        return None
    return float(times[idx[0]])


def list_image_files(frames_dir: str | Path) -> List[Path]:
    path = Path(frames_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Depth frames directory not found: {path}")
    files = [p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    files = sorted(files, key=lambda p: p.name)
    if not files:
        raise ValueError(f"No image files found in: {path}")
    return files


def _extract_numeric_token(name: str) -> Optional[str]:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", name)
    if not matches:
        return None
    return matches[-1]


def _parse_frame_token(name: str) -> Tuple[Optional[float], Optional[str]]:
    token = _extract_numeric_token(name)
    if token is None:
        return None, None
    if "." in token or "e" in token or "E" in token:
        return float(token), "time"
    return float(int(token)), "index"


def index_frames_dir(
    frames_dir: str | Path,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, List[Path]]:
    files = list_image_files(frames_dir)
    times: List[float] = []
    used_files: List[Path] = []
    skipped = 0
    for path in files:
        val, kind = _parse_frame_token(path.stem)
        if val is None or kind is None:
            skipped += 1
            continue
        if kind == "index":
            if fps is None or fps <= 0:
                raise ValueError(
                    "Depth frames appear to be indexed; provide --depth-fps to map to seconds."
                )
            t = float(val) / float(fps)
        else:
            t = float(val)
        times.append(t)
        used_files.append(path)
    if skipped:
        print(f"[WARN] Skipped {skipped} depth frames without numeric tokens.", file=sys.stderr)
    if not times:
        raise ValueError("No depth frames with numeric tokens were found.")
    order = np.argsort(times)
    times_sorted = np.asarray(times, dtype=float)[order]
    files_sorted = [used_files[i] for i in order]
    return times_sorted, files_sorted


def read_image(path: str | Path) -> np.ndarray:
    import cv2

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return img


def open_video(path: str | Path) -> Any:
    import cv2

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {p}")
    return cap


def read_frame_cv2(cap: Any, t_sec: float) -> Optional[np.ndarray]:
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def read_frame_ffmpeg(video_path: str | Path, t_sec: float) -> Optional[np.ndarray]:
    import cv2

    if shutil.which("ffmpeg") is None:
        return None
    p = Path(video_path)
    if not p.exists():
        return None
    cmd = [
        "ffmpeg",
        "-ss",
        f"{float(t_sec):.6f}",
        "-i",
        str(p),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return None
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return frame


def extract_frame(
    video_path: str | Path,
    t_sec: float,
    cap: Optional[Any] = None,
) -> Optional[np.ndarray]:
    frame = None
    if cap is not None:
        frame = read_frame_cv2(cap, t_sec)
    if frame is None:
        frame = read_frame_ffmpeg(video_path, t_sec)
    return frame
