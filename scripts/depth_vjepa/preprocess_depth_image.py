#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


_COLORMAPS = {
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "jet": cv2.COLORMAP_JET,
    "bone": cv2.COLORMAP_BONE,
    "gray": None,
}


def _collect_images(input_path: Path) -> tuple[list[Path], Path]:
    if input_path.is_file():
        return [input_path], input_path.parent
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in input_path.rglob("*") if p.suffix.lower() in exts]
    images = sorted({p.resolve() for p in images})
    return images, input_path


def _normalize_depth(depth: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    d = depth.astype(np.float32)
    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=np.float32)

    v = d[finite]
    lo = np.percentile(v, pmin)
    hi = np.percentile(v, pmax)
    if hi <= lo:
        hi = lo + 1e-6

    d_norm = (d - lo) / (hi - lo)
    d_norm = np.clip(d_norm, 0.0, 1.0)
    d_norm[~finite] = 0.0
    return d_norm


def _depth_to_color(depth_norm: np.ndarray, colormap: str) -> np.ndarray:
    d_uint8 = (depth_norm * 255.0).astype(np.uint8)
    if colormap == "gray":
        return cv2.cvtColor(d_uint8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(d_uint8, _COLORMAPS[colormap])


def _resize_if_needed(depth_map: np.ndarray, out_size: int) -> np.ndarray:
    if out_size <= 0:
        return depth_map
    h, w = depth_map.shape[:2]
    if h == out_size and w == out_size:
        return depth_map
    return cv2.resize(depth_map, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _run_inference(
    model: DepthAnything3,
    images: list[Image.Image],
    process_res: int,
    process_res_method: str,
) -> list[np.ndarray]:
    pred = model.inference(
        image=images,
        process_res=process_res,
        process_res_method=process_res_method,
        export_dir=None,
        export_format=[],
    )
    depth_list = pred.depth
    if isinstance(depth_list, np.ndarray):
        if depth_list.ndim == 3:
            return [depth_list[i] for i in range(depth_list.shape[0])]
        return [depth_list]
    if torch.is_tensor(depth_list):
        if depth_list.ndim == 3:
            return [depth_list[i].cpu().numpy() for i in range(depth_list.shape[0])]
        return [depth_list.cpu().numpy()]
    return list(depth_list)


def _save_outputs(
    img_path: Path,
    pil_img: Image.Image,
    depth_map: np.ndarray,
    out_root: Path,
    rel_root: Path,
    save_res: int,
    rgb_res: int,
    colormap: str,
    pmin: float,
    pmax: float,
    invert: bool,
    save_raw: bool,
):
    rel = img_path.relative_to(rel_root)
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = rel.stem
    rgb_path = out_dir / f"{stem}_rgb.png"
    depth_img_path = out_dir / f"{stem}_depth.png"
    depth_raw_path = out_dir / f"{stem}_depth.npz"

    depth_map = _resize_if_needed(depth_map, save_res)

    d_norm = _normalize_depth(depth_map, pmin=pmin, pmax=pmax)
    if invert:
        d_norm = 1.0 - d_norm
    d_vis = _depth_to_color(d_norm, colormap=colormap)

    cv2.imwrite(str(depth_img_path), d_vis)

    if save_raw:
        np.savez_compressed(depth_raw_path, depth=depth_map.astype(np.float16))

    if rgb_res < 0:
        rgb_res = save_res
    if rgb_res > 0:
        pil_out = pil_img.resize((rgb_res, rgb_res), resample=Image.BILINEAR)
    else:
        pil_out = pil_img
    pil_out.save(rgb_path)

    return rgb_path, depth_img_path, depth_raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Image file or folder.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--save-res", type=int, default=224, help="Depth output size. <=0 keeps model size.")
    parser.add_argument("--rgb-res", type=int, default=-1, help="-1 uses save-res, 0 keeps original size.")
    parser.add_argument("--colormap", type=str, default="turbo", choices=sorted(_COLORMAPS.keys()))
    parser.add_argument("--pmin", type=float, default=2.0)
    parser.add_argument("--pmax", type=float, default=98.0)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--save-raw", action="store_true", help="Save depth as .npz (float16).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    out_root = Path(args.out_dir).resolve()
    images, rel_root = _collect_images(input_path)

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(images)

    if args.max_images and int(args.max_images) > 0:
        images = images[: int(args.max_images)]

    if not images:
        raise RuntimeError("No images found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Loading {args.model_name} on {device}...")
    model = DepthAnything3.from_pretrained(args.model_name).to(device).eval()

    skipped = 0
    processed = 0

    pbar = tqdm(images)
    batch = []
    batch_paths = []

    for img_path in pbar:
        rel = img_path.relative_to(rel_root)
        out_dir = out_root / rel.parent
        rgb_path = out_dir / f"{rel.stem}_rgb.png"
        depth_img_path = out_dir / f"{rel.stem}_depth.png"
        depth_raw_path = out_dir / f"{rel.stem}_depth.npz"

        if not args.overwrite:
            needed = [rgb_path, depth_img_path]
            if args.save_raw:
                needed.append(depth_raw_path)
            if all(p.exists() for p in needed):
                skipped += 1
                continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            tqdm.write(f"[WARN] Failed to open {img_path}: {exc}")
            continue

        batch.append(pil_img)
        batch_paths.append(img_path)

        if len(batch) >= args.batch_size:
            depths = _run_inference(
                model,
                batch,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
            )
            if len(depths) != len(batch):
                tqdm.write("[WARN] Depth count mismatch. Skipping batch.")
            else:
                for p, img, d in zip(batch_paths, batch, depths):
                    _save_outputs(
                        p,
                        img,
                        d,
                        out_root=out_root,
                        rel_root=rel_root,
                        save_res=args.save_res,
                        rgb_res=args.rgb_res,
                        colormap=args.colormap,
                        pmin=args.pmin,
                        pmax=args.pmax,
                        invert=args.invert,
                        save_raw=args.save_raw,
                    )
                    processed += 1
            batch = []
            batch_paths = []

    if batch:
        depths = _run_inference(
            model,
            batch,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
        if len(depths) != len(batch):
            print("[WARN] Depth count mismatch on final batch.")
        else:
            for p, img, d in zip(batch_paths, batch, depths):
                _save_outputs(
                    p,
                    img,
                    d,
                    out_root=out_root,
                    rel_root=rel_root,
                    save_res=args.save_res,
                    rgb_res=args.rgb_res,
                    colormap=args.colormap,
                    pmin=args.pmin,
                    pmax=args.pmax,
                    invert=args.invert,
                    save_raw=args.save_raw,
                )
                processed += 1

    print("-" * 50)
    print("DONE")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {out_root}")
    print("-" * 50)


if __name__ == "__main__":
    main()
