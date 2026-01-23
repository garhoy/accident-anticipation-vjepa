#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de entrenamiento individual (llamado por train_batch.py).

CAMBIOS vs versión anterior:
- Limpieza de choices en argparse (solo modelos útiles)
- Mensajes más informativos durante entrenamiento
- Sin cambios en lógica de entrenamiento (ya estaba bien)
"""
from __future__ import annotations
import argparse, json, random, os, traceback, math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel

from dataset_embeddings import VJEPAFramewiseDataset, collate_pad
from dataset_tokens import DatasetTokens, collate_tokens
from models import build_model
from metrics import (
    compute_val_metrics_from_frame_logits,
    mtta_threshold_free_lit,
    tta_at_threshold,
)

# -----------------------------
# Utilidades
# -----------------------------
def _iter_pts(dirs: List[str]):
    for d in dirs:
        p = Path(d)
        if not p.exists(): 
            continue
        yield from p.glob("*.pt")

def _infer_geometry_from_sample(pt_path: Path, use_tokens: bool) -> Tuple[int, Dict[str, Any]]:
    """Infiere dimensión D y geometría de tokens desde un .pt de ejemplo."""
    data = torch.load(pt_path, map_location="cpu")
    geom = {}
    
    if use_tokens:
        if "tokens" not in data:
            raise ValueError(
                f"--use-tokens activado, pero {pt_path} no contiene 'tokens'. "
                f"Verifica que los datos tengan tokens guardados con save_tokens."
            )
        D = int(data["tokens"].shape[-1])
        meta = data.get("tokens_meta", {})
        geom = {
            "grid_h": int(meta.get("grid_h", 16)),
            "grid_w": int(meta.get("grid_w", 16)),
            "frames_per_clip": int(meta.get("frames_per_clip", 16)),
            "tubelet_size": int(meta.get("tubelet_size", 2)),
        }
        print(f"✅ Modo TOKENS: D={D}, Geometría={geom}")
    else:
        if "z_clip" not in data:
            raise ValueError(f"{pt_path} no contiene 'z_clip'.")
        D = int(data["z_clip"].shape[-1])
        print(f"✅ Modo EMBEDDINGS: D={D}")
    
    return D, geom

def print_val_metrics(
    valm: Dict[str, float], 
    mtta: float, 
    tta_r80: float,
    epoch: int, 
    ap_scope: str = "anytime"
):
    """Imprime métricas de validación de forma compacta."""
    ap_vid = float(valm.get("AP_video_lit", 0.0))
    auc_vid = float(valm.get("AUC_video_lit", 0.0))
    
    mtta_str = f"{mtta:.2f}s" if np.isfinite(mtta) else "N/A"
    tta_str = f"{tta_r80:.2f}s" if np.isfinite(tta_r80) else "N/A"
    
    print(
        f"📊 E{epoch:02d} | "
        f"AP[{ap_scope}]={ap_vid:.4f} | "
        f"AUC={auc_vid:.4f} | "
        f"mTTA={mtta_str} | "
        f"TTA@R80={tta_str} | "
    )

# -----------------------------
# VALIDACIÓN
# -----------------------------
@torch.no_grad()
def run_validation(
    model: torch.nn.Module, 
    val_ld: DataLoader, 
    horizons: List[float],
    device: torch.device, 
    ap_scope: str, 
    use_tokens: bool
):
    model.eval()
    
    logits_all, masks_all, deltas_all, times_all = [], [], [], []
    y_video_all, t_event_all = [], []

    for b in tqdm(val_ld, desc="Validación", leave=False):
        m = b["masks"].to(device)
        d = b["deltas"].to(device)
        t = b["clip_times"].to(device)
        y_vid = b["targets"].cpu().numpy()
        t_evt = b["t_events"].cpu().numpy()

        if use_tokens:
            x_tok = b["tokens"].to(device)           # (B, T_clips, L, D)
            tok_mask = b.get("token_mask", None)
            if tok_mask is not None:
                tok_mask = tok_mask.to(device)        # (B, T_clips, L)

            B, T_clips, L, D_tok = x_tok.shape
            logits_bt = []
            for t_clip in range(T_clips):
                clip_tokens = x_tok[:, t_clip, :, :]
                clip_mask   = tok_mask[:, t_clip, :] if tok_mask is not None else None
                logits_clip = model(clip_tokens, token_mask=clip_mask)
                logits_t = logits_clip[:, -1, :] if logits_clip.ndim == 3 else logits_clip
                logits_bt.append(logits_t)
            logits_b = torch.stack(logits_bt, dim=1)     # (B, T_clips, nW)
        else:
            x_emb = b["embeddings"].to(device)
            m_emb = b["masks"].to(device)
            logits_b = model(x_emb, mask=m_emb)

        logits_all.append(logits_b.cpu())
        masks_all.append(m.cpu())
        deltas_all.append(d.cpu())
        times_all.append(t.cpu())
        y_video_all.extend(y_vid.tolist())
        t_event_all.extend(t_evt.tolist())

    logits_bt_w = torch.cat(logits_all, dim=0)
    mask_bt     = torch.cat(masks_all,  dim=0)
    deltas_bt   = torch.cat(deltas_all, dim=0)
    times_bt    = torch.cat(times_all,  dim=0)
    y_video_np  = np.asarray(y_video_all)
    t_events_np = np.asarray(t_event_all)
    y_video_t   = torch.tensor(y_video_np, dtype=torch.float32)
    t_events    = torch.tensor(t_events_np, dtype=torch.float32)

    # Métricas principales (AP/AUC + umbrales ANYTIME y PRE en 'extras')
    val_metrics = compute_val_metrics_from_frame_logits(
        logits_bt_w=logits_bt_w, 
        mask_bt=mask_bt, 
        deltas_bt=deltas_bt,
        y_video_t=y_video_t, 
        horizons=horizons, 
        dt_shift=0.0,
        ap_scope=ap_scope, 
        times_bt=times_bt, 
        t_events=t_events,
    )

    # mTTA y TTA@R80 (PRE y ANYTIME)
    mtta_free_lit = float("nan")
    tta_R80_pre_lit = float("nan")
    tta_R80_any_lit = float("nan")

    try:
        probs_bt_w = torch.sigmoid(logits_bt_w)

        mtta_free_lit = mtta_threshold_free_lit(
            probs_bt_w=probs_bt_w,
            times_bt=times_bt,
            mask_bt=mask_bt,
            t_events=t_events_np,
            y_video=y_video_np,
            q_grid="paper19",
            strict_no_cross=True,
            deltas_bt=deltas_bt,
            dt_shift=0.0,
        )

        thr_R80_any = float(val_metrics["extras"]["anytime"]["thr_R80_lit"])
        thr_R80_pre = float(val_metrics["extras"]["pre"]["thr_R80_lit"])

        tta_R80_pre_lit = tta_at_threshold(
            probs_bt_w=probs_bt_w, times_bt=times_bt, mask_bt=mask_bt,
            t_events=t_events_np, thr=thr_R80_pre, y_video=y_video_np,
            strict_no_cross=True, deltas_bt=deltas_bt, dt_shift=0.0
        )
        tta_R80_any_lit = tta_at_threshold(
            probs_bt_w=probs_bt_w, times_bt=times_bt, mask_bt=mask_bt,
            t_events=t_events_np, thr=thr_R80_any, y_video=y_video_np,
            strict_no_cross=True, deltas_bt=deltas_bt, dt_shift=0.0
        )
    except Exception as e:
        print(f"⚠️  [WARN] mTTA/TTA@R80 no calculables: {e}")

    val_metrics = dict(val_metrics)
    val_metrics["mTTA_free_lit"] = float(mtta_free_lit) if np.isfinite(mtta_free_lit) else None
    val_metrics["TTA_R80_PRE"]   = float(tta_R80_pre_lit) if np.isfinite(tta_R80_pre_lit) else None
    val_metrics["TTA_R80_ANY"]   = float(tta_R80_any_lit) if np.isfinite(tta_R80_any_lit) else None
    
    # Devuelvo TTA_PRE como tercero (coherente con impresión compacta)
    return val_metrics, float(mtta_free_lit), float(tta_R80_pre_lit)



# -----------------------------
# OPTIMIZADOR / SCHEDULER / EMA
# -----------------------------
def build_param_groups(model: torch.nn.Module, wd: float, lr: float):
    """Separa parámetros con/sin weight decay (AdamW decoupled)."""
    decay, no_decay = [], []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        # Bias, LayerNorm, BatchNorm sin weight decay
        is_nd = (
            (p.ndim == 1) or 
            n.endswith(".bias") or 
            ("norm" in n.lower()) or 
            ("ln" in n.lower())
        )
        
        (no_decay if is_nd else decay).append(p)
    
    return [
        {"params": decay, "weight_decay": wd, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]

def build_cosine_warmup_scheduler(optimizer, epochs: int, warmup_epochs: int):
    """Cosine annealing con warmup lineal."""
    warmup_epochs = max(0, min(warmup_epochs, max(1, epochs - 1)))
    
    def lr_lambda(epoch):
        # Fase 1: warmup lineal
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        
        # Fase 2: cosine decay
        t = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -----------------------------
# TRAIN (AMP + ACCUM + EMA)
# -----------------------------
def train_one_epoch(
    model: torch.nn.Module, 
    train_ld: DataLoader, 
    device: torch.device,
    use_tokens: bool, 
    optimizer: torch.optim.Optimizer, 
    scaler: GradScaler,
    accum_steps: int = 1, 
    amp_enabled: bool = False,
    label_smoothing_eps: float = 0.0, 
    ema: Optional[AveragedModel] = None,
    use_pos_weight: bool = False,   # <<< NUEVO: pásalo desde main(args.pos_weight)
) -> float:
    model.train()
    loss_sum, n_steps = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_ld, desc="Entrenamiento", leave=False)

    for step, b in enumerate(pbar, start=1):
        y = b["labels_tf"].to(device)  # (B, T, nW), float
        m = b["masks"].to(device)      # (B, T), bool

        if label_smoothing_eps > 0.0:
            y = y * (1.0 - label_smoothing_eps) + 0.5 * label_smoothing_eps

        with autocast(device_type="cuda", enabled=(amp_enabled and device.type == "cuda")):
            if use_tokens:
                x_tok = b["tokens"].to(device)           # (B, T_clips, L, D)
                tok_mask = b.get("token_mask", None)
                if tok_mask is not None:
                    tok_mask = tok_mask.to(device)        # (B, T_clips, L)

                B, T_clips, L, D_tok = x_tok.shape
                logits_bt = []
                for t_clip in range(T_clips):
                    clip_tokens = x_tok[:, t_clip, :, :]  # (B, L, D)
                    clip_mask   = tok_mask[:, t_clip, :] if tok_mask is not None else None
                    logits_clip = model(clip_tokens, token_mask=clip_mask)
                    logits_t = logits_clip[:, -1, :] if logits_clip.ndim == 3 else logits_clip
                    logits_bt.append(logits_t)
                logits = torch.stack(logits_bt, dim=1)    # (B, T_clips, nW)
            else:
                x_emb = b["embeddings"].to(device)
                m_emb = b["masks"].to(device)
                logits = model(x_emb, mask=m_emb)         # (B, T, nW)

            # ----- BCE con pos_weight opcional por HORIZONTE -----
            mask_expanded = m.unsqueeze(-1).expand_as(y).float()  # (B,T,nW)
            pos_w = None
            if use_pos_weight:
                with torch.no_grad():
                    pos = (y * mask_expanded).sum(dim=(0,1)) + 1e-6   # (nW,)
                    neg = ((1.0 - y) * mask_expanded).sum(dim=(0,1)) + 1e-6
                    pos_w = (neg / pos).clamp(1.0, 100.0).to(logits.device)

            bce = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none", pos_weight=pos_w
            )
            # ----- PROMEDIO POR VÍDEO -----
            loss_b = (bce * mask_expanded).sum(dim=(1,2)) / mask_expanded.sum(dim=(1,2)).clamp_min(1.0)
            loss   = loss_b.mean()

        loss_scaled = loss / max(1, accum_steps)
        scaler.scale(loss_scaled).backward()

        if step % max(1, accum_steps) == 0:
            scaler.unscale_(optimizer)
            clip = 5.0 if use_tokens else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update_parameters(model)

        loss_sum += float(loss.item())
        n_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return loss_sum / max(n_steps, 1)



# -----------------------------
# MAIN
# -----------------------------
def main(args):
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar config
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Dispositivo: {device}")

    # Semillas para reproducibilidad
    random.seed(args.run_idx)
    np.random.seed(args.run_idx)
    torch.manual_seed(args.run_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_idx)

    # Inferir dimensión y geometría desde un sample
    print("\n🔍 Infiriendo dimensiones desde sample...")
    first_pt = next(_iter_pts(args.train_dirs))
    D, geom_kwargs = _infer_geometry_from_sample(first_pt, use_tokens=args.use_tokens)

    # Datasets
    post_window_s = 0.0  # protocolo "end"
    
    if args.use_tokens:
        train_ds = DatasetTokens(
            args.train_dirs, args.window_s, post_window_s, 
            args.horizons, video_ids=None
        )
        val_ds = DatasetTokens(
            args.val_dirs, args.window_s, post_window_s, 
            args.horizons, video_ids=None
        )
        collate_fn = collate_tokens
    else:
        train_ds = VJEPAFramewiseDataset(
            args.train_dirs, args.window_s, post_window_s, 
            args.horizons, video_ids=None
        )
        val_ds = VJEPAFramewiseDataset(
            args.val_dirs, args.window_s, post_window_s, 
            args.horizons, video_ids=None
        )
        collate_fn = collate_pad

    print(f"📚 Dataset Train: {len(train_ds)} vídeos | Val: {len(val_ds)} vídeos")

    # Sampler balanceado (opcional)
    sampler: Optional[WeightedRandomSampler] = None
    if args.balanced_sampler:
        print("⚖️  Usando WeightedRandomSampler (balanceo 50/50)...")
        weights = train_ds.get_sampler_weights()
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_fn, 
        drop_last=False,
    )
    
    val_bs = args.batch_size if args.use_tokens else args.batch_size * 2
    val_loader = DataLoader(
        val_ds, 
        batch_size=val_bs, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=collate_fn
    )

    # Construir modelo
    try:
        model_kw = json.loads(args.model_kw)
    except Exception as e:
        raise SystemExit(f"❌ --model-kw no es JSON válido: {e}") from e

    print(f"\n🏗️  Construyendo modelo: {args.model}")
    if model_kw:
        print(f"   Config: {json.dumps(model_kw, indent=2)}")
    
    model_kw.pop("model_type", None)
    model = build_model(
        model_type=args.model, 
        embed_dim=D, 
        n_windows=len(args.horizons),
        **geom_kwargs, 
        **model_kw
     ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros entrenables: {n_params/1e6:.2f}M")

    # Optimizer (AdamW con weight decay decoupled)
    base_lr = 1e-4
    wd = 1e-3
    label_smoothing_eps = 0.0
    
    # Transformers suelen beneficiarse de LR más alto
    if args.model == "transformer":
        base_lr = 2e-4
    

    if args.use_tokens:
        base_lr = 3e-4

    param_groups = build_param_groups(model, wd=wd, lr=base_lr)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    print(f"⚙️  Optimizer: AdamW(lr={base_lr}, wd={wd})")

    # Scheduler
    scheduler = None
    if args.cosine:
        scheduler = build_cosine_warmup_scheduler(
            optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs
        )
        print(f"📈 Scheduler: CosineAnnealing + Warmup ({args.warmup_epochs} épocas)")
    elif getattr(args, "plateau_lr", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.plateau_factor, 
            patience=(
                args.plateau_patience 
                if args.plateau_patience >= 0 
                else max(1, args.early_stop_patience // 2)
            ),
            threshold=args.early_stop_min_delta, 
            threshold_mode="abs", 
            cooldown=0, 
            min_lr=0.0, 
            eps=1e-8
        )
        print(f"📈 Scheduler: ReduceLROnPlateau(factor={args.plateau_factor})")


    scaler = GradScaler('cuda', enabled=args.amp and device.type=="cuda")


    # reemplaza la línea del scaler por:
    # scaler = GradScaler(device="cuda", enabled=(args.amp and device.type == "cuda"))



    accum_steps = max(1, args.accum_steps)
    
    ema = None
    if args.ema:
        ema = AveragedModel(
            model, 
            avg_fn=lambda avg, cur, num: (
                avg * args.ema_decay + (1.0 - args.ema_decay) * cur
            )
        )
        ema.to(device)
        print(f"🔄 EMA activado (decay={args.ema_decay})")

    # Early stopping
    best_ap = -1.0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []
    ap_scope = args.ap_scope
    min_epochs = max(1, args.min_epochs)
    es_patience = max(1, args.early_stop_patience)
    es_delta = float(args.early_stop_min_delta)

    print("\n" + "="*70)
    print("🚀 INICIO DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Modelo: {args.model} | Run: {args.run_idx} | Épocas: {args.epochs}")
    print(f"Output: {out_dir}")
    print(f"Protocolo: {ap_scope.upper()}")
    print("="*70 + "\n")

    for epoch in range(1, args.epochs + 1):
        # Entrenar
        train_loss = train_one_epoch(
            model, train_loader, device, args.use_tokens, optimizer,
            scaler=scaler, accum_steps=accum_steps, amp_enabled=args.amp,
            label_smoothing_eps=0.0, ema=ema, use_pos_weight=args.pos_weight  # <<<
        )

        # Validar (con EMA si está activado)
        eval_model = ema.module if ema is not None else model
        
        try:
            valm, mtta_free_lit, tta_R80_lit = run_validation(
                model=eval_model, 
                val_ld=val_loader, 
                horizons=args.horizons,
                device=device, 
                ap_scope=ap_scope, 
                use_tokens=args.use_tokens,
            )
        except Exception:
            print("\n❌ [ERROR] Excepción en validación:")
            traceback.print_exc()
            raise

        print_val_metrics(valm, mtta_free_lit, tta_R80_lit, epoch, ap_scope)

        # Guardar historial
        ap_curr = float(valm.get("AP_video_lit", -1.0))
        history.append({
            "epoch": epoch,
            **{
                k: float(v) 
                for k, v in valm.items() 
                if isinstance(v, (int, float)) or v is None
            },
            "mTTA_free_lit": (
                float(valm.get("mTTA_free_lit")) 
                if valm.get("mTTA_free_lit") is not None 
                else None
            ),
            "TTA_R80": (
                float(valm.get("TTA_R80")) 
                if valm.get("TTA_R80") is not None 
                else None
            ),
            "train_loss": float(train_loss),
        })

        # Scheduler step
        if scheduler is not None:
            if args.cosine:
                scheduler.step()
            else:
                scheduler.step(ap_curr)

        # Guardar checkpoint si mejora
        improved = (ap_curr - best_ap) > es_delta
        
        if improved:
            best_ap = ap_curr
            epochs_no_improve = 0
            
            torch.save({
                "model": (ema.module if ema is not None else model).state_dict(),
                "model_raw": model.state_dict(),
                "ema_active": (ema is not None),
                "config": {
                    "model_type": args.model, 
                    "embed_dim": D, 
                    "horizons": args.horizons, 
                    "window_s": args.window_s,
                    "decision_point": "end", 
                    "event_align": "clip_end", 
                    "mode": "batch",
                    "n_params": n_params, 
                    "ap_scope": ap_scope, 
                    "use_tokens": args.use_tokens,
                    **geom_kwargs, 
                    **model_kw,
                },
                "val_metrics": valm,
                "epoch": epoch,
            }, out_dir / "best_model.pt")
            
            print(f"💾 [SAVED] Mejor modelo (AP={ap_curr:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epoch >= min_epochs and epochs_no_improve >= es_patience:
            print(
                f"\n⏹️  [EARLY STOP] Sin mejora durante {epochs_no_improve} épocas "
                f"(Δ≤{es_delta}). Mejor AP[{ap_scope}] = {best_ap:.4f}."
            )
            break

    # Guardar historial completo
    with (out_dir / "val_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETO")
    print("="*70)
    print(f"Modelo: {args.model} (run {args.run_idx})")
    print(f"Mejor AP[{ap_scope}]: {best_ap:.4f}")
    print(f"Resultados en: {out_dir}")
    print("="*70 + "\n")

# -----------------------------
# ARGPARSE
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("Entrenamiento individual (embeddings o tokens)")
    
    # Rutas
    p.add_argument("--train-dirs", nargs="+", required=True)
    p.add_argument("--val-dirs", nargs="+", required=True)
    p.add_argument("--out-root", type=str, required=True)
    
    # Modelo
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "gru", "tcn", "transformer",  # embeddings
            "badas", "causal_film_tcn", "temporal_attn", "deformable_event"  # tokens
        ],
        help="Arquitectura a entrenar"
    )
    p.add_argument("--model-kw", type=str, default="{}", help="JSON con hiperparámetros del modelo")
    
    # Hiperparámetros
    p.add_argument("--window-s", type=float, default=5.0)
    p.add_argument("--horizons", type=float, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    
    # Flags
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument("--use-tokens", action="store_true")
    
    # Evaluación
    p.add_argument("--ap-scope", type=str, default="anytime", choices=["anytime", "pre"])
    
    # Early stopping
    p.add_argument("--min-epochs", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--early-stop-min-delta", type=float, default=0.001)
    
    # Schedulers
    p.add_argument("--plateau-lr", action="store_true")
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-patience", type=int, default=-1)
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--warmup-epochs", type=int, default=1)
    
    # Optimización avanzada
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--amp", action="store_true", help="Activar mixed precision (AMP)")
    p.add_argument("--ema", action="store_true", help="Activar EMA")
    p.add_argument("--ema-decay", type=float, default=0.999)
    # Imbalance (mantén sampler + pos_weight)
    p.add_argument("--pos-weight", action="store_true", help="Activar pos_weight dinámico por horizonte")
    
    # Otros
    p.add_argument("--run-idx", type=int, default=0)
    
    return p.parse_args()

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    args = get_args()
    main(args)
