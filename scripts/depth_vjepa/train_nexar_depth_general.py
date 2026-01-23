#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nexar_depth_robust_v2.py

Entrenamiento Depth Multimodal (RGB + Depth) con paridad EXACTA al Fine-Tuning de V-JEPA.

Características:
  - Sampler Clonado: Hard Negatives + Jitter [min, max] idéntico a V-JEPA FT.
  - Objetivo Idéntico: Loss sobre Max-Pooling temporal (Anytime Hard).
  - Ingeniería: AMP (Mixed Precision), Resume completo, Debug Mode.
  - Arquitectura: V-JEPA (Frozen/Unfrozen) + Depth3D + Transformer Head.

Uso:
  python train_nexar_depth_robust_v2.py --train-csv ... --val-csv ... \
    --video-root ... --depth-root ... \
    --train-jitter-min 0.0 --train-jitter-max 2.0
"""

import os
import sys
import csv
import argparse
import random
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video

from transformers import AutoVideoProcessor, AutoConfig, VJEPA2Model

import wandb

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# OJO: adapta este import al path real de tu repo
from models import build_model

from thesis.utils.csv_utils import float_or_none, get_balanced_subset


# ==============================================================================
# 1. UTILIDADES
# ==============================================================================

def _resolve_video_path(videos_root: Path, vid: str) -> Optional[Path]:
    valid_exts = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"}
    matches = list(videos_root.rglob(f"{vid}.*"))
    for p in matches:
        if p.suffix in valid_exts:
            return p
    return None

def _resolve_depth_path(depth_root: Path, vid: str) -> Optional[Path]:
    p = depth_root / f"{vid}.npz"
    if p.exists():
        return p
    found = list(depth_root.rglob(f"{vid}.npz"))
    return found[0] if found else None

def _load_depth_clip_nearest(npz_path: Path, frame_indices: np.ndarray) -> Optional[np.ndarray]:
    try:
        data = np.load(str(npz_path))
        d_stack = data["depth"]       
        d_indices = data["frame_idx"] 
        if d_stack.shape[0] == 0: return None

        diffs = np.abs(d_indices[None, :] - frame_indices[:, None])
        nearest = diffs.argmin(axis=1)         
        clip = d_stack[nearest]                
        clip = clip[..., None].astype(np.float32)  
        return clip
    except Exception:
        return None


# ==============================================================================
# 2. DATASET (Lógica Clonada de V-JEPA FT)
# ==============================================================================

class NexarDepthRobustDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        data_rows: Optional[List] = None,
        videos_root: Path = None,
        depth_root: Path = None,
        frames_per_clip: int = 16,
        hist_s: float = 5.0,
        train_mode: bool = True,
        hard_neg_prob: float = 0.5,
        jitter_range: Tuple[float, float] = (0.0, 2.0), # (min, max) positivo
        safe_margin_s: float = 2.0
    ):
        super().__init__()
        self.videos_root = videos_root
        self.depth_root = depth_root
        self.frames_per_clip = int(frames_per_clip)
        self.hist_s = float(hist_s)
        
        # Params Hard Mode
        self.train_mode = train_mode
        self.hard_neg_prob = hard_neg_prob
        self.jitter_range = jitter_range
        self.safe_margin_s = safe_margin_s

        # Carga de datos
        rows_to_process = []
        if data_rows: rows_to_process = data_rows
        elif csv_path:
            with csv_path.open("r", encoding="utf-8") as f:
                rows_to_process = list(csv.DictReader(f))
        
        self.samples = []
        for row in rows_to_process:
            vid = row["id"].strip()
            if not vid: continue
            try: target = int(float(row["target"]))
            except: continue
            if target not in (0, 1): continue

            t_event = float_or_none(row.get("time_of_event", ""))
            vpath = _resolve_video_path(self.videos_root, vid)
            if vpath is None: continue
            dpath = _resolve_depth_path(self.depth_root, vid)

            self.samples.append({
                "id": vid, "vpath": vpath, "dpath": dpath, 
                "target": target, "time_of_event": t_event
            })

        print(f"[DATASET] {len(self.samples)} vídeos. Mode={'TRAIN (Hard)' if train_mode else 'VAL'}")

    def __len__(self) -> int: return len(self.samples)

    def _get_indices_robust(self, T, fps, t_event, target_video):
        """Lógica de muestreo EXACTA a train_nexar_vjepa_ft.py"""
        duration = T / max(fps, 1e-6)
        final_label = target_video

        # --- VAL / TEST ---
        if not self.train_mode:
            if target_video == 1 and t_event is not None:
                center_sec = t_event # FT usa t_event directo en val (final del evento)
            else:
                center_sec = 0.5 * duration
            return self._sec_to_idx(center_sec, duration, fps, T), final_label

        # --- TRAIN ---
        use_positive_sample = False
        
        # 1. Hard Negative Logic
        if target_video == 1 and t_event is not None:
            if torch.rand(1).item() > self.hard_neg_prob:
                use_positive_sample = True 
                final_label = 1
            else:
                use_positive_sample = False
                final_label = 0 
        else:
            use_positive_sample = False
            final_label = 0

        # 2. Jitter Logic (Positivo [min, max])
        if use_positive_sample:
            low, high = self.jitter_range
            offset = torch.empty(1).uniform_(low, high).item()
            center_sec = t_event + offset # Causal: Ventana termina un poco después del evento
        else:
            # Mining de fondo
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
                center_sec = t_event if target_video==1 else 0.5*duration
                if target_video==1: final_label = 1

        return self._sec_to_idx(center_sec, duration, fps, T), final_label

    def _sec_to_idx(self, center_sec, duration, fps, T):
        """Mapeo tiempo -> índices (Clonado de FT)"""
        start_sec = max(0.0, center_sec - self.hist_s)
        end_sec = min(duration, center_sec)
        
        # Fallback idéntico al FT
        if end_sec <= start_sec: 
            start_sec = 0.0
            end_sec = duration

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

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        try:
            video, _, info = read_video(str(sample["vpath"]), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except:
            video = torch.zeros((self.frames_per_clip, 224, 224, 3), dtype=torch.uint8)
            fps = 30.0

        if video.numel() == 0:
            video = torch.zeros((self.frames_per_clip, 224, 224, 3), dtype=torch.uint8)

        idxs, label_clip = self._get_indices_robust(video.shape[0], fps, sample["time_of_event"], sample["target"])
        
        rgb_clip = video[idxs].numpy()
        
        dpath = sample["dpath"]
        depth_clip = None
        if dpath: depth_clip = _load_depth_clip_nearest(dpath, idxs.numpy())
        if depth_clip is None:
            depth_clip = np.zeros((self.frames_per_clip, 224, 224, 1), dtype=np.float32)

        return rgb_clip, depth_clip, label_clip


def make_collate_fn(processor: AutoVideoProcessor):
    def collate_fn(batch):
        rgbs, depths, labels = zip(*batch)
        inputs = processor(list(rgbs), return_tensors="pt")

        # depths: lista de [T, H, W, 1] float16/float32
        d_np = np.stack(depths, axis=0)          # [B, T, H, W, 1]
        d_t = torch.from_numpy(d_np).float()     # [B, T, H, W, 1]

        # Quitamos el canal dummy -> [B, T, H, W]
        d_t = d_t[..., 0]

        # 1) Clamp físico: 0 a 100 metros
        d_t = torch.clamp(d_t, min=0.0, max=150.0)

        # 2) Log-escala + reescalado
        d_t = torch.log1p(d_t) / 5.0

        # 3) Añadimos canal: [B, 1, T, H, W]
        d_t = d_t.unsqueeze(1).contiguous()

        labels_t = torch.tensor(labels, dtype=torch.float32)
        return inputs, d_t, labels_t

    return collate_fn


# ==============================================================================
# 3. MODELO 
# ==============================================================================

class Depth3DPerFrame(nn.Module):
    def __init__(self, depth_dim: int = 128, base_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(base_channels), nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(base_channels * 2), nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(base_channels * 4), nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(base_channels * 4, depth_dim)

    def forward(self, depth_btchw: torch.Tensor) -> torch.Tensor:
        feat = self.net(depth_btchw).mean(dim=[3, 4]).permute(0, 2, 1)
        return self.proj(feat)


class VJEPA2DepthTransformerBinary(nn.Module):
    def __init__(self, hf_repo: str, depth_dim: int = 128, unfreeze_blocks: int = 0, encoder_ckpt: Optional[str] = None):
        super().__init__()
        self.vjepa2 = VJEPA2Model.from_pretrained(hf_repo)
        cfg = self.vjepa2.config
        self.embed_dim = cfg.hidden_size
        self.frames_per_clip = getattr(cfg, "frames_per_clip", 16)
        crop, patch = getattr(cfg, "crop_size", 256), getattr(cfg, "patch_size", 16)
        self.grid = crop // patch
        self.tubelet = cfg.tubelet_size
        self.steps = self.frames_per_clip // self.tubelet

        if encoder_ckpt:
            sd = torch.load(encoder_ckpt, map_location="cpu")
            if "model_state" in sd: sd = sd["model_state"]
            self.vjepa2.load_state_dict(sd, strict=False)

        for p in self.vjepa2.parameters(): p.requires_grad = False
        if unfreeze_blocks > 0: self._unfreeze_blocks(unfreeze_blocks)

        self.depth_net = Depth3DPerFrame(depth_dim=depth_dim)
        fused_dim = self.embed_dim + depth_dim

        self.temporal_head = build_model(
            model_type="transformer",
            embed_dim=fused_dim,
            n_windows=1,
            d_model=384,
            n_heads=6,
            num_layers=4,
            ff_dim=1536,
            dropout=0.10,
            pe_dropout=0.05,
            attn_dropout=0.05,
            causal=True,
        )

    def _unfreeze_blocks(self, n: int):
        blocks = getattr(self.vjepa2.encoder, "layers", getattr(self.vjepa2.encoder, "layer", None))
        if blocks:
            for i, block in enumerate(blocks):
                if i >= len(blocks) - n:
                    for p in block.parameters(): p.requires_grad = True

    def _rgb_embeddings_per_frame(self, pv):
        out = self.vjepa2(pixel_values_videos=pv, output_hidden_states=False)
        tokens = out.last_hidden_state
        B, L, D = tokens.shape
        if L != self.steps * self.grid**2: tokens = tokens[:, 1:, :] # Guardarse todos los tokens no quitar el 256 de dim
        x = tokens.view(B, self.steps, self.grid**2, D).mean(dim=2) # ! quitamos mean 16 x 16 x 256 x 1024
        x = F.interpolate(x.transpose(1, 2), size=self.frames_per_clip, mode="linear", align_corners=False).transpose(1, 2)
        return x

    def forward(self, pixel_values_videos, depth_videos, return_frame_logits=False):
        z_rgb = self._rgb_embeddings_per_frame(pixel_values_videos)
        z_depth = self.depth_net(depth_videos)
        if z_depth.shape[1] != z_rgb.shape[1]:
            z_depth = F.interpolate(z_depth.transpose(1, 2), size=z_rgb.shape[1], mode="linear").transpose(1, 2)
        
        z_fused = torch.cat([z_rgb, z_depth], dim=-1)
        B, T, _ = z_fused.shape
        logits_bt = self.temporal_head(z_fused, mask=torch.ones(B, T, device=z_fused.device, dtype=torch.bool)).squeeze(-1)
        
        if return_frame_logits: return logits_bt
        return logits_bt.max(dim=1).values


# ==============================================================================
# 4. TRAINING LOOPS (AMP + Max Loss)
# ==============================================================================

def train_one_epoch(model, dataloader, device, optimizer, criterion, scaler, epoch, clip_norm):
    model.train()
    total_loss, total_samples = 0.0, 0
    agg_tp = agg_fp = agg_tn = agg_fn = 0

    pbar = tqdm(dataloader, desc=f"Train E{epoch:02d}", unit="bt") if tqdm else dataloader

    for inputs, depth, labels in pbar:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        depth = depth.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits_bt = model(inputs["pixel_values_videos"], depth, return_frame_logits=True)
            # OBJETIVO IDENTICO A FT: Loss sobre el maximo del clip
            logits_clip = logits_bt.max(dim=1).values
            loss = criterion(logits_clip, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        B = labels.size(0)
        total_loss += loss.item() * B
        total_samples += B

        with torch.no_grad():
            preds = (torch.sigmoid(logits_clip) >= 0.5).long()
            y = labels.long()
            agg_tp += ((preds==1)&(y==1)).sum().item()
            agg_fp += ((preds==1)&(y==0)).sum().item()
            agg_tn += ((preds==0)&(y==0)).sum().item()
            agg_fn += ((preds==0)&(y==1)).sum().item()

        if tqdm:
            acc = (agg_tp+agg_tn)/total_samples
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.1%}")

    mean_loss = total_loss / max(1, total_samples)
    acc, f1, prec, rec = calculate_global_metrics(agg_tp, agg_fp, agg_tn, agg_fn)
    return mean_loss, acc*100.0,f1

@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_samples = 0.0, 0
    agg_tp = agg_fp = agg_tn = agg_fn = 0

    for inputs, depth, labels in dataloader:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        depth = depth.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_bt = model(inputs["pixel_values_videos"], depth, return_frame_logits=True)
        # OBJETIVO IDENTICO A FT
        logits_clip = logits_bt.max(dim=1).values
        loss = criterion(logits_clip, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        preds = (torch.sigmoid(logits_clip) >= 0.5).long()
        y = labels.long()
        agg_tp += ((preds==1)&(y==1)).sum().item()
        agg_fp += ((preds==1)&(y==0)).sum().item()
        agg_tn += ((preds==0)&(y==0)).sum().item()
        agg_fn += ((preds==0)&(y==1)).sum().item()

    mean_loss = total_loss / max(1, total_samples)
    acc, f1, prec, rec = calculate_global_metrics(agg_tp, agg_fp, agg_tn, agg_fn)
    return mean_loss, acc * 100.0, f1, prec, rec, (agg_tp, agg_fp, agg_tn, agg_fn)

def calculate_global_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(1, total)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, f1, prec, rec


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--depth-root", required=True)
    ap.add_argument("--out-dir", default="checkpoints_depth_robust")
    
    # Model
    ap.add_argument("--hf-repo", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--depth-dim", type=int, default=128)
    
    # Train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    
    # Robustness (Paridad con FT)
    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--hard-neg-prob", type=float, default=0.5)
    ap.add_argument("--train-jitter-min", type=float, default=0.0, help="Jitter Min (seg)")
    ap.add_argument("--train-jitter-max", type=float, default=2.0, help="Jitter Max (seg)")
    
    # Utils
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None)

    args = ap.parse_args()

    # Safety Check
    if args.train_jitter_max >= args.hist_s:
        print(f"[ERR] Jitter Max ({args.train_jitter_max}) >= Hist ({args.hist_s}). Evento fuera de rango.")
        sys.exit(1)

    # W&B
    wandb.init(project="nexar_tfm_experiments", name=args.run_name, config=vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    fpc = getattr(AutoConfig.from_pretrained(args.hf_repo), "frames_per_clip", 16)

    # Dataset Selection
    jitter_range = (args.train_jitter_min, args.train_jitter_max)
    
    if args.debug:
        print("[DEBUG] Usando subset en memoria...")
        train_rows = get_balanced_subset(Path(args.train_csv), 80)
        val_rows = get_balanced_subset(Path(args.val_csv), 30)
        train_ds = NexarDepthRobustDataset(data_rows=train_rows, videos_root=Path(args.video_root), depth_root=Path(args.depth_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=True, hard_neg_prob=args.hard_neg_prob, jitter_range=jitter_range)
        val_ds = NexarDepthRobustDataset(data_rows=val_rows, videos_root=Path(args.video_root), depth_root=Path(args.depth_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=False)
    else:
        train_ds = NexarDepthRobustDataset(csv_path=Path(args.train_csv), videos_root=Path(args.video_root), depth_root=Path(args.depth_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=True, hard_neg_prob=args.hard_neg_prob, jitter_range=jitter_range)
        val_ds = NexarDepthRobustDataset(csv_path=Path(args.val_csv), videos_root=Path(args.video_root), depth_root=Path(args.depth_root), frames_per_clip=fpc, hist_s=args.hist_s, train_mode=False)

    collate = make_collate_fn(processor)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, pin_memory=True)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=True)

    model = VJEPA2DepthTransformerBinary(args.hf_repo, args.depth_dim, args.unfreeze_blocks, args.encoder_ckpt).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    scaler = torch.cuda.amp.GradScaler()

    # Resume Logic
    start_epoch = 1
    best_val_f1 = 0.0
    
    if args.resume and os.path.isfile(args.resume):
        print(f"[RESUME] Cargando {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt: scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("val_f1", 0.0)
    elif args.init_from and os.path.isfile(args.init_from):
        print(f"[INIT] Cargando pesos de {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device)
        st = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(st, strict=False)

    # Loop
    for epoch in range(start_epoch, args.epochs + 1):
        t_loss, t_acc, t_f1 = train_one_epoch(model, train_ld, device, optimizer, criterion, scaler, epoch, args.grad_clip)
        v_loss, v_acc, v_f1, v_prec, v_rec, (tp, fp, tn, fn) = evaluate(model, val_ld, device, criterion)

        print(f"[E{epoch:02d}] T_Loss:{t_loss:.3f} | V_Loss:{v_loss:.3f} F1:{v_f1:.3f} (P:{v_prec:.2f}/R:{v_rec:.2f})")
        print(f"       CM: [TN:{tn} FP:{fp}] [FN:{fn} TP:{tp}]")

        wandb.log({
            "epoch": epoch, 
            "train/loss": t_loss, 
            "train/acc": t_acc, 
            "train/f1": t_f1,  # <--- Ahora sí existe t_f1 y se guardará en W&B
            "val/loss": v_loss, "val/acc": v_acc, "val/f1": v_f1,
            "val/prec": v_prec, "val/rec": v_rec,
            "val/cm/tp": tp, "val/cm/fp": fp, "val/cm/tn": tn, "val/cm/fn": fn,
            "best_val_f1": max(best_val_f1, v_f1)
        })

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_f1": v_f1,
                "args": vars(args)
            }, Path(args.out_dir) / "best_model_robust.pt")
            print(f"       ✅ Saved Best F1: {v_f1:.3f}")

    wandb.finish()

if __name__ == "__main__":
    main()
