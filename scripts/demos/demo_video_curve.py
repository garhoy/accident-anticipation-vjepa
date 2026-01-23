#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_video_curve.py

Script de visualización para V-JEPA 2 Fine-Tuned.
Genera una gráfica de probabilidad de accidente a lo largo del tiempo.

Uso:
  python demo_video_curve.py --csv path/to/train.csv --video-root path/to/videos --checkpoint path/to/model.pt
"""

import os
import sys
import csv
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.io import read_video
from transformers import AutoConfig, AutoVideoProcessor, VJEPA2Model

# ------------------------------------------------------------------------------
# 1. IMPORTS DE MODELOS (Standalone)
# ------------------------------------------------------------------------------
try:
    from models import (
        BADASAttentiveProbe,
        DeformableEventQueryHead,
        TransformerPerFrame,
    )
except ImportError:
    print("[WARN] models.py no encontrado. Asegúrate de ejecutar desde la raíz.")

# ------------------------------------------------------------------------------
# 2. CLASE DEL MODELO (Para cargar checkpoint sin dependencias)
# ------------------------------------------------------------------------------
class VJEPA2TemporalBinary(nn.Module):
    def __init__(self, hf_repo, head_type="badas_attn", n_windows=1, unfreeze_blocks=0, encoder_ckpt=None):
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
            if L != (self.steps * self.grid**2): tokens = tokens[:, 1:, :]
            seq = tokens.view(B, self.steps, self.grid**2, D).mean(dim=2)
            mask = torch.ones(seq.shape[0], seq.shape[1], dtype=torch.bool, device=seq.device)
            logits_seq = self.head(seq, mask=mask)
            logits_t = nn.functional.interpolate(logits_seq.transpose(1,2), size=self.frames_per_clip, mode='linear').transpose(1,2)

        return logits_t.max(dim=1).values.squeeze(-1)

# ------------------------------------------------------------------------------
# 3. FUNCIONES DE UTILIDAD
# ------------------------------------------------------------------------------
def get_random_accident(csv_path):
    """Busca un vídeo aleatorio con target=1 en el CSV."""
    print(f"[CSV] Buscando accidentes en: {csv_path}")
    candidates = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row['target'])) == 1:
                    candidates.append(row)
            except: continue
    
    if not candidates:
        raise ValueError("No se encontraron accidentes (target=1) en el CSV.")
    
    chosen = random.choice(candidates)
    print(f"[CSV] ¡Video encontrado! ID: {chosen['id']} (Evento en t={chosen.get('time_of_event', '???')}s)")
    return chosen

def find_video_file(video_root, vid_id):
    """Busca el archivo de vídeo con varias extensiones."""
    for ext in ['.mp4', '.MP4', '.mov', '.MOV', '.mkv', '.avi']:
        path = Path(video_root) / f"{vid_id}{ext}"
        if path.exists():
            return path
    return None

def analyze_video(video_path, t_event_gt, model, processor, device, output_img="curva_accidente.png"):
    print(f"[VIDEO] Leyendo: {video_path} ...")
    video, _, info = read_video(str(video_path), pts_unit="sec")
    fps = info.get("video_fps", 30.0)
    T = video.shape[0]
    duration = T / fps
    print(f"[VIDEO] Duración: {duration:.2f}s | FPS: {fps}")

    # Sliding Window fino para la gráfica
    hist_s = 5.0
    stride_s = 0.2 # Alta resolución temporal para la gráfica
    
    centers = np.arange(hist_s, duration, stride_s)
    probs = []
    times = []

    model.eval()
    print("[INFER] Generando curva de probabilidad...")
    
    with torch.no_grad():
        # Procesar batch a batch para ir rápido
        batch_size = 8
        clips_batch = []
        batch_times = []
        
        for t in centers:
            # Extraer clip [t-5, t]
            end_sec = min(duration, t)
            start_sec = max(0.0, end_sec - hist_s)
            
            start_idx = int(start_sec * fps)
            end_idx = int(end_sec * fps)
            
            # Asegurar 16 frames (sampleo simple)
            indices = torch.linspace(start_idx, end_idx-1, 16).long().clamp(0, T-1)
            clip = video[indices] # (16, H, W, C)
            
            clips_batch.append(clip.numpy())
            batch_times.append(t)
            
            if len(clips_batch) >= batch_size:
                inputs = processor(clips_batch, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs)
                p = torch.sigmoid(logits).cpu().numpy().flatten()
                probs.extend(p)
                times.extend(batch_times)
                clips_batch, batch_times = [], []
        
        # Procesar el resto
        if clips_batch:
            inputs = processor(clips_batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs)
            p = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(p)
            times.extend(batch_times)

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Curva de probabilidad
    plt.plot(times, probs, color='#d62728', linewidth=2.5, label='Probabilidad Modelo')
    plt.fill_between(times, probs, color='#d62728', alpha=0.1)
    
    # Línea de GT (si existe)
    if t_event_gt is not None:
        plt.axvline(x=t_event_gt, color='#2ca02c', linestyle='--', linewidth=2, label=f'Accidente Real (t={t_event_gt}s)')
        plt.text(t_event_gt, 1.02, " IMPACTO", color='#2ca02c', fontweight='bold')

    # Umbral
    plt.axhline(y=0.5, color='gray', linestyle=':', label='Umbral Detección (0.5)')
    
    plt.title(f'Detección de Accidente: {video_path.name}', fontsize=14)
    plt.xlabel('Tiempo (segundos)', fontsize=12)
    plt.ylabel('Probabilidad de Accidente', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.savefig(output_img, dpi=150)
    print(f"[PLOT] Gráfica guardada en: {output_img}")
    print(f"[PLOT] Score Máximo alcanzado: {max(probs):.4f} en t={times[np.argmax(probs)]:.2f}s")

# ------------------------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Ruta al train.csv")
    parser.add_argument("--video-root", type=str, required=True, help="Carpeta de videos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Modelo .pt")
    parser.add_argument("--id", type=str, default=None, help="ID específico (opcional). Si no, coge random.")
    parser.add_argument("--head-type", type=str, default="transformer")
    parser.add_argument("--out-img", type=str, default="curva_accidente.png")
    args = parser.parse_args()

    # 1. Elegir vídeo
    if args.id:
        print(f"Buscando ID específico: {args.id}")
        # Buscar manual en CSV para sacar el tiempo
        target_row = None
        with open(args.csv, 'r') as f:
            for r in csv.DictReader(f):
                if r['id'] == args.id:
                    target_row = r
                    break
        if not target_row:
            print("[WARN] ID no encontrado en CSV, no tendré tiempo de evento.")
            target_row = {"id": args.id, "time_of_event": None}
    else:
        target_row = get_random_accident(args.csv)

    vid_id = target_row['id']
    try:
        t_event = float(target_row['time_of_event']) if target_row.get('time_of_event') else None
    except: t_event = None

    # 2. Buscar archivo
    vpath = find_video_file(args.video_root, vid_id)
    if not vpath:
        print(f"[ERR] Archivo de vídeo no encontrado para ID: {vid_id}")
        sys.exit(1)

    # 3. Cargar Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Cargando modelo: {args.checkpoint} en {device}")
    
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = VJEPA2TemporalBinary("facebook/vjepa2-vitl-fpc16-256-ssv2", head_type=args.head_type)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    
    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")

    # 4. Analizar
    analyze_video(vpath, t_event, model, processor, device, args.out_img)

if __name__ == "__main__":
    main()