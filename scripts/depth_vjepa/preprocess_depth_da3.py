#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image

from depth_anything_3.api import DepthAnything3


def process_video_optimized(
    video_path: Path,
    save_path: Path,
    model: DepthAnything3,
    device: torch.device,
    target_fps: int = 5,     # profundidad a 5 fps
    save_res: int = 224,     # mapas 224x224
    batch_size: int = 8,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] No puedo abrir vídeo: {video_path}")
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0

    # cada cuántos frames cogemos uno
    frame_step = max(1, int(round(orig_fps / target_fps)))

    depth_frames = []
    frame_indices = []

    frame_idx = 0
    batch_images = []
    batch_indices = []

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                batch_images.append(pil_img)
                batch_indices.append(frame_idx)

                if len(batch_images) >= batch_size:
                    pred = model.inference(
                        image=batch_images,
                        process_res=504,
                        process_res_method="upper_bound_resize",
                        export_dir=None,
                        export_format=[],
                    )
                    # pred.depth: [N, H, W] float32 (CPU)
                    for d_map in pred.depth:
                        d_small = cv2.resize(
                            d_map,
                            (save_res, save_res),
                            interpolation=cv2.INTER_AREA,
                        )
                        depth_frames.append(d_small.astype(np.float16))

                    frame_indices.extend(batch_indices)
                    batch_images = []
                    batch_indices = []

            frame_idx += 1

        # último batch
        if batch_images:
            pred = model.inference(
                image=batch_images,
                process_res=504,
                process_res_method="upper_bound_resize",
                export_dir=None,
                export_format=[],
            )
            for d_map in pred.depth:
                # CORREGIDO: Usar INTER_AREA también aquí para consistencia
                d_small = cv2.resize(
                    d_map,
                    (save_res, save_res),
                    interpolation=cv2.INTER_AREA, 
                )
                depth_frames.append(d_small.astype(np.float16))
            frame_indices.extend(batch_indices)

    cap.release()

    if len(depth_frames) == 0:
        print(f"[WARN] No se han generado mapas de profundidad para {video_path}")
        return None

    depth_stack = np.stack(depth_frames, axis=0)  # [T, H, W]
    frame_idx_arr = np.asarray(frame_indices, dtype=np.int32)

    # --- CALCULO DE ESTADÍSTICAS ---
    stats = {
        "video": video_path.name,
        "tensor_shape": depth_stack.shape,  # (T, H, W)
        "dtype": str(depth_stack.dtype),
        "min_depth": float(depth_stack.min()),
        "max_depth": float(depth_stack.max()),
        "orig_fps": round(orig_fps, 2),
        "file_size_mb": depth_stack.nbytes / (1024 * 1024)
    }

    np.savez_compressed(
        save_path,
        depth=depth_stack,      # float16, [T, 224, 224]
        frame_idx=frame_idx_arr,  # int32, [T]
        fps=np.float32(orig_fps),
    )
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
    )
    parser.add_argument("--target-fps", type=int, default=5)
    parser.add_argument("--save-res", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-videos", type=int, default=0, help="0 = todos; si >0 limita a N vídeos")
    parser.add_argument("--shuffle", action="store_true", help="baraja el orden de vídeos antes de limitar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="recalcula depth aunque el .npz ya exista")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Cargando {args.model_name} en {device}...")

    model = DepthAnything3.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    root = Path(args.video_root).resolve()
    out = Path(args.out_root).resolve()
    exts = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}
    videos = [p for p in root.rglob("*") if p.suffix in exts]
    videos = sorted({v.resolve() for v in videos})

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(videos)

    if args.max_videos and int(args.max_videos) > 0:
        videos = videos[: int(args.max_videos)]

    print(
        f"[INFO] Encontrados {len(videos)} vídeos. "
        f"Salida: {out} | fps={args.target_fps} | res={args.save_res}"
    )

    # Acumuladores globales (solo para los vídeos realmente procesados en este run)
    total_frames_processed = 0
    total_size_mb = 0.0
    n_processed = 0
    n_skipped_existing = 0

    # Usamos tqdm
    pbar = tqdm(videos)
    for v in pbar:
        rel = v.relative_to(root)
        dest = (out / rel).with_suffix(".npz")
        
        # Si ya existe, podemos saltar, pero no imprimimos stats
        if dest.exists() and (not args.overwrite):
            n_skipped_existing += 1
            continue

        stats = process_video_optimized(
            v,
            dest,
            model,
            device=device,
            target_fps=args.target_fps,
            save_res=args.save_res,
            batch_size=args.batch_size,
        )

        if stats:
            n_processed += 1
            total_frames_processed += stats["tensor_shape"][0]
            total_size_mb += stats["file_size_mb"]
            
            # Mensaje de resumen por video (usando tqdm.write para no romper la barra)
            # Mostramos: Nombre, Shape [T, H, W], Rango de profundidad (Min-Max)
            msg = (
                f"✅ {stats['video']} | "
                f"Shape: {stats['tensor_shape']} | "
                f"Range: [{stats['min_depth']:.1f}, {stats['max_depth']:.1f}] | "
                f"{stats['file_size_mb']:.2f} MB"
            )
            tqdm.write(msg)
    
    print("-" * 50)
    print("🚀 PROCESO FINALIZADO")
    print(f"Vídeos procesados: {n_processed}")
    print(f"Vídeos saltados (ya existían): {n_skipped_existing}")
    print(f"Total Frames (Tensors) generados en este run: {total_frames_processed}")
    print(f"Total Size generado (aprox raw) en este run: {total_size_mb / 1024:.2f} GB")
    if n_processed == 0 and n_skipped_existing > 0:
        print("[INFO] No se recalculó nada porque los .npz ya existían. Usa --overwrite o cambia --out-root si quieres regenerar.")
    print("-" * 50)

if __name__ == "__main__":
    main()
