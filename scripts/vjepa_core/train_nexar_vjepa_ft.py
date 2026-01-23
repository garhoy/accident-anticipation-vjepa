#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nexar_vjepa_ft.py

Fine-tuning V-JEPA 2 para Detección de Accidentes (Nexar).

CARACTERÍSTICAS:
  - Debug Mode (--debug): Coge subsets aleatorios de los CSVs reales en memoria.
  - Hard Negative Mining: Enseña que "no chocar" != accidente.
  - Safe Jitter: El evento siempre cae dentro del clip.
  - Robustez: Class Balancing (pos_weight) y Gradient Clipping.

USO DEBUG (Rápido):
  python train_nexar_vjepa_ft.py --train-csv ... --val-csv ... --video-root ... --debug

USO REAL:
  python train_nexar_vjepa_ft.py --train-csv ... --val-csv ... --video-root ...
"""

import os
import sys
import csv
import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video

from transformers import AutoVideoProcessor, AutoConfig, VJEPA2Model
import wandb
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Intentar importar tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -------------------------
# IMPORTS LOCALES
# -------------------------
try:
    from models import (
        BADASAttentiveProbe,
        DeformableEventQueryHead,
        TransformerPerFrame,
    )
except ImportError:
    print("[ERR] No se encuentra models.py. Asegúrate de tener las cabezas definidas.")
    sys.exit(1)

from thesis.utils.csv_utils import float_or_none, get_balanced_subset


# ==============================================================================
# 2. DATASET (Híbrido: Acepta CSV o Lista)
# ==============================================================================

class NexarTemporalFTDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[Path] = None,   # Opción A: Ruta archivo
        data_rows: Optional[List] = None,  # Opción B: Lista en memoria (Debug)
        videos_root: Path = None,
        frames_per_clip: int = 16,
        hist_s: float = 5.0,
        max_videos: int = -1,
        train_mode: bool = True,
        jitter_range: Tuple[float, float] = (0.0, 2.0),
        hard_neg_prob: float = 0.5,
        safe_margin_s: float = 2.0,
    ):
        super().__init__()
        if videos_root is None: raise ValueError("videos_root es obligatorio")
        if not csv_path and not data_rows: raise ValueError("Debes pasar csv_path O data_rows")

        self.videos_root = videos_root
        self.frames_per_clip = int(frames_per_clip)
        self.hist_s = float(hist_s)
        self.train_mode = bool(train_mode)
        self.jitter_range = jitter_range
        self.hard_neg_prob = float(hard_neg_prob)
        self.safe_margin_s = float(safe_margin_s)

        # --- LÓGICA DE CARGA ---
        rows_to_process = []
        source_name = "Unknown"

        if data_rows is not None:
            rows_to_process = data_rows
            source_name = "In-Memory List (Debug)"
        elif csv_path is not None:
            if not csv_path.exists(): raise FileNotFoundError(csv_path)
            with csv_path.open("r", encoding="utf-8") as f:
                rows_to_process = list(csv.DictReader(f))
            source_name = str(csv_path)

        self.samples = []
        for row in rows_to_process:
            vid = row["id"].strip()
            if vid == "": continue
            try: target_csv = int(float(row["target"]))
            except: continue
            if target_csv not in (0, 1): continue

            t_event = float_or_none(row.get("time_of_event", ""))
            vpath = self._resolve_path(vid)
            if vpath is None: continue

            self.samples.append({
                "id": vid, "path": vpath, "target_csv": target_csv, "time_of_event": t_event
            })
            if max_videos > 0 and len(self.samples) >= max_videos: break

        if len(self.samples) == 0:
            raise RuntimeError(f"Ninguna muestra válida en {source_name}")
        
        mode_str = "TRAIN (HardNegs ON)" if self.train_mode else "VAL/TEST"
        print(f"[DATASET] {len(self.samples)} vídeos cargados desde {source_name}. Mode: {mode_str}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, vid: str) -> Optional[Path]:
        for ext in [".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"]:
            p = self.videos_root / f"{vid}{ext}"
            if p.exists(): return p
        return None

    def _sample_indices(self, T, fps, t_event, target_video) -> Tuple[torch.Tensor, int]:
        duration = T / max(fps, 1e-6)
        final_label = target_video
        
        # --- VAL / TEST ---
        if not self.train_mode:
            if target_video == 1 and t_event is not None:
                center_sec = t_event
            else:
                center_sec = 0.5 * duration
            return self._sec_to_idx(center_sec, duration, fps, T), final_label

        # --- TRAIN ---
        use_positive_sample = False
        
        # 1. Decisión Hard Negative
        if target_video == 1 and t_event is not None:
            if torch.rand(1).item() > self.hard_neg_prob:
                use_positive_sample = True 
                final_label = 1
            else:
                use_positive_sample = False # Fondo
                final_label = 0 
        else:
            use_positive_sample = False
            final_label = 0

        # 2. Jitter
        if use_positive_sample:
            low, high = self.jitter_range
            offset = torch.empty(1).uniform_(low, high).item()
            center_sec = t_event + offset
        else:
            # Hard Negative Mining (buscar hueco)
            valid_intervals = []
            if t_event is not None:
                end_z1 = t_event - self.safe_margin_s
                if end_z1 > self.hist_s: valid_intervals.append((self.hist_s, end_z1))
                start_z2 = t_event + self.safe_margin_s + self.hist_s
                if start_z2 < duration: valid_intervals.append((start_z2, duration))
            else:
                if duration > self.hist_s: valid_intervals.append((self.hist_s, duration))

            if valid_intervals:
                idx = torch.randint(len(valid_intervals), (1,)).item()
                s, e = valid_intervals[idx]
                center_sec = torch.empty(1).uniform_(s, e).item()
            else:
                # Fallback
                center_sec = t_event if target_video==1 else 0.5*duration
                if target_video==1: final_label = 1 # Revertir a positivo

        return self._sec_to_idx(center_sec, duration, fps, T), final_label

    def _sec_to_idx(self, center_sec, duration, fps, T):
        start_sec = max(0.0, center_sec - self.hist_s)
        end_sec = min(duration, center_sec)
        if end_sec <= start_sec: start_sec = 0.0; end_sec = duration
        start_idx = int(round(start_sec * fps))
        end_idx = int(round(end_sec * fps))
        start_idx = max(0, min(start_idx, T - 1))
        end_idx = max(start_idx, min(end_idx, T - 1))

        if start_idx == end_idx:
            idx = torch.full((self.frames_per_clip,), start_idx, dtype=torch.long)
        else:
            idx = torch.linspace(start_idx, end_idx, steps=self.frames_per_clip).long()
            idx = torch.clamp(idx, 0, T - 1)
        return idx

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        sample = self.samples[idx]
        try:
            video, _, info = read_video(str(sample["path"]), pts_unit="sec")
        except Exception:
            video = torch.zeros((self.frames_per_clip, 224, 224, 3), dtype=torch.uint8)
            info = {}

        if video.numel() == 0:
             video = torch.zeros((self.frames_per_clip, 224, 224, 3), dtype=torch.uint8)
        
        T = video.shape[0]
        fps = float(info.get("video_fps", 30.0))
        idxs, label_clip = self._sample_indices(T, fps, sample["time_of_event"], sample["target_csv"])
        return video[idxs].numpy(), label_clip

def make_collate_fn(processor: AutoVideoProcessor):
    def collate_fn(batch):
        videos_np, labels = zip(*batch)
        inputs = processor(list(videos_np), return_tensors="pt")
        labels_t = torch.tensor(labels, dtype=torch.float32)
        return inputs, labels_t
    return collate_fn

# ==============================================================================
# 3. MODELO
# ==============================================================================

class VJEPA2TemporalBinary(nn.Module):
    def __init__(self, hf_repo, head_type="badas_attn", n_windows=1, unfreeze_blocks=0, encoder_ckpt=None):
        super().__init__()
        self.head_type = head_type
        print(f"[MODEL] Backbone: {hf_repo}")
        self.vjepa2 = VJEPA2Model.from_pretrained(hf_repo)

        if encoder_ckpt:
            print(f"[MODEL] Encoder Externo: {encoder_ckpt}")
            sd = torch.load(encoder_ckpt, map_location="cpu")
            if "model_state" in sd: sd = sd["model_state"]
            sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}
            self.vjepa2.load_state_dict(sd, strict=False)

        cfg = self.vjepa2.config
        self.embed_dim = cfg.hidden_size
        self.frames_per_clip = getattr(cfg, "frames_per_clip", 16)
        self.grid = getattr(cfg, "crop_size", 256) // getattr(cfg, "patch_size", 16)
        self.tubelet = cfg.tubelet_size
        self.steps = self.frames_per_clip // self.tubelet

        for p in self.vjepa2.parameters(): p.requires_grad = False
        self._init_head()
        if unfreeze_blocks > 0: self._unfreeze_blocks(unfreeze_blocks)

    def _init_head(self):
        if self.head_type == "badas_attn":
            self.head = BADASAttentiveProbe(self.embed_dim, 1, self.grid, self.grid, self.frames_per_clip, self.tubelet)
        elif self.head_type == "deformable_event":
            self.head = DeformableEventQueryHead(self.embed_dim, 1, self.grid, self.grid, self.frames_per_clip, self.tubelet)
        else:
            self.head = TransformerPerFrame(self.embed_dim, 1, d_model=384)
        for p in self.head.parameters(): p.requires_grad = True

    def _unfreeze_blocks(self, n):
        print(f"[FT] Descongelando últimos {n} bloques.")
        tot = len(self.vjepa2.encoder.layer)
        for i, l in enumerate(self.vjepa2.encoder.layer):
            if i >= tot - n:
                for p in l.parameters(): p.requires_grad = True

    def forward(self, pixel_values_videos, **kwargs):
        out = self.vjepa2(pixel_values_videos=pixel_values_videos, output_hidden_states=False)
        tokens = out.last_hidden_state
        
        if self.head_type in ["badas_attn", "deformable_event"]:
            logits_t = self.head(tokens)
        else:
            # Simple Transformer Logic
            B, L, D = tokens.shape
            if L != (self.steps * self.grid**2): tokens = tokens[:, 1:, :]
            seq = tokens.view(B, self.steps, self.grid**2, D).mean(dim=2)
            mask = torch.ones(seq.shape[0], seq.shape[1], dtype=torch.bool, device=seq.device)
            logits_seq = self.head(seq, mask=mask)
            logits_t = nn.functional.interpolate(logits_seq.transpose(1,2), size=self.frames_per_clip, mode='linear').transpose(1,2)

        return logits_t.max(dim=1).values.squeeze(-1)

# ==============================================================================
# 4. LOOPS
# ==============================================================================

def train_one_epoch(model, dataloader, device, optimizer, criterion, scaler, epoch, clip_norm):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    iterator = tqdm(dataloader, desc=f"Train E{epoch:02d}", unit="bt") if tqdm else dataloader

    for inputs, labels in iterator:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(**inputs)
            if logits.dim() > 1: logits = logits.squeeze()
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        B = labels.size(0)
        loss_sum += loss.item() * B
        total += B
        with torch.no_grad():
            correct += ((torch.sigmoid(logits) >= 0.5).long() == labels.long()).sum().item()
        
        if tqdm: iterator.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
    return loss_sum/total, 100.0*correct/total

@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    loss_sum, tp, fp, tn, fn, total = 0.0, 0, 0, 0, 0, 0
    for inputs, labels in dataloader:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)
        logits = model(**inputs)
        if logits.dim() > 1: logits = logits.squeeze()
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        total += labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        y = labels.long()
        tp += ((preds==1)&(y==1)).sum().item(); fp += ((preds==1)&(y==0)).sum().item()
        tn += ((preds==0)&(y==0)).sum().item(); fn += ((preds==0)&(y==1)).sum().item()

    acc = 100.0*(tp+tn)/max(1, total)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return loss_sum/total, acc, f1, prec, rec, (tp, fp, tn, fn)

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--val-csv", type=str, required=True)
    ap.add_argument("--video-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="checkpoints_ft_robust")
    
    # Model Params
    ap.add_argument("--hf-repo", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--head-type", type=str, default="badas_attn")
    ap.add_argument("--encoder-ckpt", type=str, default=None)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    
    # Training Params
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    
    # Robustness Params
    ap.add_argument("--hard-neg-prob", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--train-jitter-min", type=float, default=0.0)
    ap.add_argument("--train-jitter-max", type=float, default=2.0)
    
    # DEBUG & RESUME
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--debug", action="store_true", help="Usar subsets aleatorios de los CSV reales en memoria")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--resume", type=str, default=None, help="Ruta al checkpoint .pt para continuar entrenando")
    ap.add_argument("--init-from", type=str, default=None,
                    help="Checkpoint .pt para inicializar SOLO los pesos del modelo (sin optimizer/scaler)")
    args = ap.parse_args()

    if args.resume and args.init_from:
        print("[ERR] No puedes usar --resume y --init-from a la vez.")
        sys.exit(1)

    wandb.init(
        project="nexar_vjepa_vitL_ft",  # nombre que quieras
        name=args.run_name,
        config=vars(args),
    )
    # Safety Check
    if args.train_jitter_max >= args.hist_s:
        print(f"[ERR] Jitter Max ({args.train_jitter_max}) >= Hist ({args.hist_s}). Evento fuera de rango."); sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    fpc = getattr(AutoConfig.from_pretrained(args.hf_repo), "frames_per_clip", 16)

    # --- SELECCIÓN DE DATOS ---
    if args.debug:
        print("\n!!! MODO DEBUG ACTIVADO: Usando subsets de memoria !!!")
        train_rows = get_balanced_subset(Path(args.train_csv), n_total=80) 
        val_rows   = get_balanced_subset(Path(args.val_csv), n_total=30)   
        train_ds = NexarTemporalFTDataset(data_rows=train_rows, videos_root=Path(args.video_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=True, jitter_range=(args.train_jitter_min, args.train_jitter_max), hard_neg_prob=args.hard_neg_prob)
        val_ds   = NexarTemporalFTDataset(data_rows=val_rows, videos_root=Path(args.video_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=False)
    else:
        train_ds = NexarTemporalFTDataset(csv_path=Path(args.train_csv), videos_root=Path(args.video_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=True, jitter_range=(args.train_jitter_min, args.train_jitter_max), hard_neg_prob=args.hard_neg_prob)
        val_ds   = NexarTemporalFTDataset(csv_path=Path(args.val_csv), videos_root=Path(args.video_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=False)

    collate = make_collate_fn(processor)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_ld   = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=True)

    print(f"[INIT] Device: {device} | PosWeight: {args.pos_weight} | HardNeg: {args.hard_neg_prob}")
    
    model = VJEPA2TemporalBinary(args.hf_repo, args.head_type, unfreeze_blocks=args.unfreeze_blocks, encoder_ckpt=args.encoder_ckpt).to(device)
    
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "params_total": n_total,
        "params_trainable": n_train,
    }, allow_val_change=True)

    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    scaler = torch.cuda.amp.GradScaler()

    # ==========================================
    # LÓGICA DE RESUME (BLINDADA)
    # ==========================================
    start_epoch = 1
    best_val_f1 = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n[RESUME] Cargando checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 1. Cargar Pesos del Modelo
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"], strict=True)
            else:
                model.load_state_dict(checkpoint, strict=True)
            
            # 2. Cargar Optimizador
            if "optimizer_state" in checkpoint:
                print("[RESUME] Restaurando estado del Optimizador.")
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                print("[RESUME] WARN: No se encontró 'optimizer_state'. Se iniciará fresco (reset momentum).")

            # 3. Cargar Scaler (CON SAFETY CHECK)
            if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
                print("[RESUME] Restaurando estado del Scaler.")
                scaler.load_state_dict(checkpoint["scaler_state"])
            else:
                print("[RESUME] WARN: 'scaler_state' no encontrado o es None. Scaler fresco.")

            # 4. Recuperar época y mejor F1
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"[RESUME] Continuando desde la época {start_epoch}")
            
            if "val_f1" in checkpoint:
                best_val_f1 = checkpoint["val_f1"]
                print(f"[RESUME] Mejor F1 previo recuperado: {best_val_f1:.4f}")
            else:
                print("[RESUME] WARN: No se encontró 'val_f1' previo. Se asume 0.0.")
        else:
            print(f"[ERR] No se encuentra el archivo: {args.resume}")
            sys.exit(1)

    elif args.init_from:
        if os.path.isfile(args.init_from):
            print(f"\n[INIT-FROM] Cargando SOLO pesos del modelo desde: {args.init_from}")
            checkpoint = torch.load(args.init_from, map_location=device)
            state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
            # Cargamos solo pesos del modelo (backbone + head frozen entrenada)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[INIT-FROM] WARN: Pesos faltantes al cargar state_dict: {len(missing)}")
            if unexpected:
                print(f"[INIT-FROM] WARN: Claves inesperadas al cargar state_dict: {len(unexpected)}")

            # OJO: NO tocamos optimizer ni scaler → se quedan nuevos, con el LR que has pasado por CLI
            start_epoch = 1
            best_val_f1 = 0.0  # nuevo mejor para esta fase
            print("[INIT-FROM] Optimizador y scaler se mantienen FRESCOS (LR nuevo).")
        else:
            print(f"[ERR] No se encuentra el archivo: {args.init_from}")
            sys.exit(1)
    # ==========================================

    # SANITY CHECK DE ÉPOCAS
    if start_epoch > args.epochs:
        print(f"[INFO] start_epoch ({start_epoch}) > args.epochs ({args.epochs}). Nada que entrenar.")
        print(f"       Aumenta --epochs si quieres seguir entrenando.")
        return

    # Bucle de entrenamiento
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        t_loss, t_acc = train_one_epoch(
            model, train_ld, device, optimizer, criterion, scaler, epoch, args.grad_clip
        )
        v_loss, v_acc, v_f1, v_prec, v_rec, (tp, fp, tn, fn) = evaluate(
            model, val_ld, device, criterion
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[E{epoch:02d}] T_L:{t_loss:.3f} A:{t_acc:.1f}% | "
            f"V_L:{v_loss:.3f} F1:{v_f1:.3f} (P:{v_prec:.2f}/R:{v_rec:.2f})"
        )
        print(f"       CM: [TN:{tn} FP:{fp}] [FN:{fn} TP:{tp}]")

        # -------- WANDB LOG --------
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": t_loss,
                "train/acc": t_acc,
                "val/loss": v_loss,
                "val/acc": v_acc,
                "val/f1": v_f1,
                "val/prec": v_prec,
                "val/rec": v_rec,
                "val/tn": tn,
                "val/fp": fp,
                "val/fn": fn,
                "val/tp": tp,
                "best/val_f1": max(best_val_f1, v_f1),  # antes de actualizar best_val_f1
                "time/epoch_sec": epoch_time,
                "lr": current_lr,
            },
            step=epoch,
        )
        # ---------------------------

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            save_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_f1": v_f1,
                "args": vars(args)
            }
            torch.save(save_dict, Path(args.out_dir)/"best_robust_model.pt")
            print(f"       [CKPT] Saved NEW Best F1: {v_f1:.3f}")
        else:
            print(
                f"       [INFO] F1 ({v_f1:.3f}) no supera al mejor histórico ({best_val_f1:.3f}). "
                "No se guarda."
            )


    wandb.finish()




if __name__ == "__main__":
    main()
