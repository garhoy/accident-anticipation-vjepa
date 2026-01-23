#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_nexar_vjepa_ft.py (VERSIÓN FINAL DEFINITIVA v3)

Evaluación para V-JEPA 2 (RGB Only).
Compatible 100% con métricas de Depth (TTA, mTTA, Recall Before Event).

Correciones v3:
  - FIX CRÍTICO: Forward robusto. Evita colapso de batch si dims son [B,T].
  - UX: Imprime el tamaño del modelo y nº de parámetros.
  - CSV con nombre único basado en el checkpoint.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from transformers import AutoConfig, AutoVideoProcessor, VJEPA2Model

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Intentamos importar models, si falla pasamos (usará defaults si es necesario)
try:
    from models import (
        BADASAttentiveProbe,
        DeformableEventQueryHead,
        TransformerPerFrame,
    )
except ImportError:
    pass

# Asegurar imports desde src/ (thesis.*) aunque ejecutes el script sin instalar paquete
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis.utils.csv_utils import float_or_none

# ------------------------------------------------------------------------------
# 1. CLASE DEL MODELO (Forward Robusto)
# ------------------------------------------------------------------------------
class VJEPA2TemporalBinary(nn.Module):
    def __init__(
        self,
        hf_repo: str,
        head_type: str = "badas_attn",
        n_windows: int = 1,
    ):
        super().__init__()
        self.head_type = head_type
        self.n_windows = int(n_windows)
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
            self.head = BADASAttentiveProbe(
                embed_dim=self.embed_dim, n_windows=self.n_windows,
                grid_h=self.grid, grid_w=self.grid, frames_per_clip=self.frames_per_clip,
                tubelet_size=self.tubelet, M=12, d=64, n_heads=16, num_attn_layers=4, dropout=0.0
            )
        elif self.head_type == "deformable_event":
            self.head = DeformableEventQueryHead(
                embed_dim=self.embed_dim, n_windows=self.n_windows,
                grid_h=self.grid, grid_w=self.grid, frames_per_clip=self.frames_per_clip,
                tubelet_size=self.tubelet, n_queries=12, num_points=12, num_levels=2, num_layers=3, d_model=256
            )
        else:
            self.head = TransformerPerFrame(
                embed_dim=self.embed_dim, n_windows=self.n_windows, d_model=384, n_heads=6, num_layers=4, ff_dim=1536
            )

    def forward(self, pixel_values_videos, **kwargs):
        out = self.vjepa2(pixel_values_videos=pixel_values_videos, output_hidden_states=False)
        tokens = out.last_hidden_state
        
        # Obtener logits temporales [B, T] o [B, T, 1]
        if self.head_type in ["badas_attn", "deformable_event"]:
            logits_t = self.head(tokens)
        else:
            B, L, D = tokens.shape
            if L != (self.steps * self.grid**2): tokens = tokens[:, 1:, :]
            seq = tokens.view(B, self.steps, self.grid**2, D).mean(dim=2)
            mask = torch.ones(seq.shape[0], seq.shape[1], dtype=torch.bool, device=seq.device)
            logits_seq = self.head(seq, mask=mask)
            logits_t = nn.functional.interpolate(logits_seq.transpose(1,2), size=self.frames_per_clip, mode='linear').transpose(1,2)

        # --- FIX ROBUSTO ---
        # Aseguramos formato [B, T] eliminando la dimensión de clase si es 1
        if logits_t.dim() == 3 and logits_t.shape[-1] == 1:
            logits_t = logits_t.squeeze(-1)
        
        # Agregación ANYTIME: Max over time -> [B]
        # Ahora es seguro: siempre colapsamos dim=1 (tiempo), manteniendo dim=0 (batch)
        logits_video = logits_t.max(dim=1).values
        
        return logits_video

# ------------------------------------------------------------------------------
# 2. UTILIDADES MÉTRICAS & TTA
# ------------------------------------------------------------------------------
def average_precision_np(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = (y_true > 0).astype(np.int32)
    if np.sum(y_true) == 0: return 0.0
    order = np.argsort(-y_scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / np.sum(y)
    idx_pos = np.where(y == 1)[0]
    if idx_pos.size == 0: return 0.0
    recall_prev = np.concatenate(([0.0], recall[idx_pos[:-1]]))
    return float(np.sum((recall[idx_pos] - recall_prev) * precision[idx_pos]))

def roc_auc_np(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2: return 0.5
        return roc_auc_score(y_true, y_scores)
    except ImportError:
        return 0.5

def binary_metrics(y_true, y_scores, thr=0.5):
    preds = (y_scores >= thr).astype(int)
    tp = ((preds==1) & (y_true==1)).sum()
    fp = ((preds==1) & (y_true==0)).sum()
    tn = ((preds==0) & (y_true==0)).sum()
    fn = ((preds==0) & (y_true==1)).sum()
    
    acc = (tp+tn) / max(1, tp+fp+tn+fn)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def get_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def compute_tta_for_video(window_times: List[float], clip_probs: List[float], time_of_event: float, thr: float) -> Tuple[float, bool]:
    if time_of_event is None or time_of_event < 0: return 0.0, False
    if not window_times or not clip_probs: return 0.0, False
    times = np.asarray(window_times, dtype=np.float32)
    probs = np.asarray(clip_probs, dtype=np.float32)
    mask = (probs >= thr) & (times <= time_of_event)
    if not mask.any(): return 0.0, False
    t_detect = float(times[mask].min())
    tta = max(time_of_event - t_detect, 0.0)
    return tta, True



def find_threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.8) -> float:
    """
    Devuelve el umbral mínimo tal que recall >= target_recall.
    Si no se alcanza, devuelve 0.5 como fallback.
    """
    y_true = (y_true > 0).astype(np.int32)
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.5

    order = np.argsort(-y_score)              # scores descendente
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    tp = np.cumsum(y_sorted)
    recall = tp / max(1, n_pos)

    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return 0.5

    return float(scores_sorted[idx[0]])

# ------------------------------------------------------------------------------
# 3. LECTURA DE CSV
# ------------------------------------------------------------------------------
def read_solution_csv(csv_path: Path) -> List[Dict]:
    out = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("id", "").strip(): continue
            try: t = int(float(row["target"]))
            except: continue
            tev = float_or_none(row.get("time_of_event", ""))
            out.append({"id": row["id"].strip(), "target": t, "usage": row.get("Usage", "").lower(), "time_of_event": tev})
    return out

# ------------------------------------------------------------------------------
# 4. EVALUACIÓN (SLIDING WINDOW + TTA)
# ------------------------------------------------------------------------------
def evaluate_split(
    split_name: str,
    rows: List[Dict],
    model: nn.Module,
    processor: AutoVideoProcessor,
    fpc: int,
    hist_s: float,
    stride_s: float,
    device: torch.device,
    video_dir: Path,
    agg: str,
    top_k: int,
    batch_size_infer: int,
    tta_thr: float,
    run_tag: str
):
    print(f"\n📊 Evaluando {split_name.upper()} ({len(rows)} vídeos)...")
    
    targets, scores, ids = [], [], []
    tta_values, n_pos_with_event, n_pos_hits_before = [], 0, 0
    missing, durations, infer_times, predictions_list = 0, [], [], []
    
    # Para TTA@R80
    pos_window_times_list: List[List[float]] = []
    pos_clip_probs_list: List[List[float]] = []
    pos_time_of_event_list: List[float] = []

    iterator = tqdm(rows, desc=split_name, unit="vid") if tqdm else rows

    for r in iterator:
        vid, target = r["id"], r["target"]
        t_event = r.get("time_of_event", None)
        vpath = None
        for ext in [".mp4", ".MP4", ".mov", ".MOV", ".mkv"]:
            p = video_dir / f"{vid}{ext}"
            if p.exists():
                vpath = p
                break
        
        if vpath is None:
            missing += 1
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        try:
            video, _, info = read_video(str(vpath), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
        except Exception:
            missing += 1
            continue

        if video.numel() == 0:
            missing += 1
            continue

        T = video.shape[0]
        duration = T / max(fps, 1e-6)
        durations.append(duration)

        centers = []
        if duration <= hist_s:
            centers = [0.5 * duration]
        else:
            n_steps = int(np.floor((duration - hist_s) / stride_s)) + 1
            centers = [hist_s + i * stride_s for i in range(max(1, n_steps))]
            if centers[-1] < duration - 0.25:
                centers.append(duration)

        clips_np, window_times = [], []
        for c in centers:
            end_sec = min(duration, c)
            start_sec = max(0.0, end_sec - hist_s)
            start_idx = max(0, min(int(start_sec * fps), T - 1))
            end_idx = max(start_idx, min(int(end_sec * fps), T - 1))
            if start_idx == end_idx:
                idx = torch.full((fpc,), start_idx, dtype=torch.long)
            else:
                idx = torch.linspace(start_idx, end_idx, steps=fpc).long().clamp(0, T - 1)
            clips_np.append(video[idx].numpy())
            window_times.append(end_sec)

        clip_probs = []
        with torch.no_grad():
            for i in range(0, len(clips_np), batch_size_infer):
                batch_clips = clips_np[i : i + batch_size_infer]
                inputs = processor(batch_clips, return_tensors="pt")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                probs = torch.sigmoid(model(**inputs)).cpu().numpy()
                clip_probs.extend(probs.flatten().tolist())
        
        clip_probs = np.array(clip_probs, dtype=np.float32)
        if len(clip_probs) == 0:
            vid_score = 0.0
        elif agg == "max":
            vid_score = float(np.max(clip_probs))
        elif agg == "mean":
            vid_score = float(np.mean(clip_probs))
        elif agg == "topk":
            k = min(top_k, len(clip_probs))
            vid_score = float(np.mean(np.sort(clip_probs)[-k:])) if k > 0 else 0.0
        else:
            vid_score = float(np.max(clip_probs))

        scores.append(vid_score)
        targets.append(target)
        ids.append(vid)
        predictions_list.append({"id": vid, "target": target, "score": f"{vid_score:.6f}"})

        if (target == 1) and (t_event is not None):
            n_pos_with_event += 1
            tta, hit = compute_tta_for_video(
                window_times, clip_probs.tolist(), float(t_event), tta_thr
            )
            if hit:
                n_pos_hits_before += 1
            tta_values.append(tta)

            # Guardar para TTA@R80
            pos_window_times_list.append(window_times)
            pos_clip_probs_list.append(clip_probs.tolist())
            pos_time_of_event_list.append(float(t_event))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_times.append(time.perf_counter() - t_start)

    if len(scores) == 0:
        return {}
    
    # --- Guardado CSV con nombre robusto ---
    out_csv = f"preds_{run_tag}_{split_name}_{agg}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "target", "score"])
        w.writeheader()
        w.writerows(predictions_list)
    print(f"   -> Predicciones: {out_csv}")

    y_true = np.array(targets, dtype=np.int32)
    y_score = np.array(scores, dtype=np.float32)

    ap = average_precision_np(y_true, y_score)
    auc = roc_auc_np(y_true, y_score)
    m = binary_metrics(y_true, y_score, thr=0.5)

    avg_inf = np.mean(infer_times) if infer_times else 0.0
    avg_dur = np.mean(durations) if durations else 0.0
    
    if tta_values and n_pos_with_event > 0:
        mtta = float(np.mean(tta_values))
        rec_before = n_pos_hits_before / max(1, n_pos_with_event)
    else:
        mtta, rec_before = float("nan"), 0.0

    # ---------- Bloque R@80 ----------
    thr_r80 = find_threshold_for_recall(y_true, y_score, target_recall=0.8)
    m_r80 = binary_metrics(y_true, y_score, thr=thr_r80)

    # mTTA@R80 y Recall_before_event@R80
    if pos_window_times_list and len(pos_window_times_list) == len(pos_time_of_event_list):
        tta_vals_r80: List[float] = []
        hits_before_r80 = 0
        for w_times, c_probs, tev in zip(
            pos_window_times_list, pos_clip_probs_list, pos_time_of_event_list
        ):
            tta_r, hit_r = compute_tta_for_video(
                w_times, c_probs, tev, thr_r80
            )
            if hit_r:
                hits_before_r80 += 1
            tta_vals_r80.append(tta_r)
        mtta_r80 = float(np.mean(tta_vals_r80)) if tta_vals_r80 else float("nan")
        rec_before_r80 = hits_before_r80 / max(1, len(tta_vals_r80))
    else:
        mtta_r80 = float("nan")
        rec_before_r80 = 0.0

    return {
        "eval": len(scores),
        "miss": missing,
        "ap": ap,
        "auc": auc,
        "acc": m["acc"],
        "prec": m["prec"],
        "rec": m["rec"],
        "f1": m["f1"],
        "cm": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        "infer_ms": avg_inf * 1000.0,
        "xrt": avg_dur / avg_inf if avg_inf > 0 else 0.0,
        "mtta": mtta,
        "recall_before_event": rec_before,
        "mtta_thr": tta_thr,

        # Métricas @R80
        "thr_r80": thr_r80,
        "acc_r80": m_r80["acc"],
        "prec_r80": m_r80["prec"],
        "rec_r80": m_r80["rec"],
        "f1_r80": m_r80["f1"],
        "mtta_r80": mtta_r80,
        "recall_before_event_r80": rec_before_r80,
    }

# ------------------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--hf-repo", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--split", type=str, default="both", choices=["public", "private", "both"])
    ap.add_argument("--stride-s", type=float, default=0.5)
    ap.add_argument("--hist-s", type=float, default=5.0)
    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean", "topk"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--batch-size-infer", type=int, default=16)
    ap.add_argument("--tta-thr", type=float, default=0.5)
    ap.add_argument("--head-type", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--solution-csv", type=str, required=True)
    ap.add_argument("--video-root", type=str, required=True)

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Extraer nombre del run para el CSV (evita overwrite)
    run_tag = Path(args.checkpoint).parent.name
    print(f"[INIT] Run Tag detectado: {run_tag}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    head_type = args.head_type or ckpt_args.get("head_type", "badas_attn")
    
    model = VJEPA2TemporalBinary(args.hf_repo, head_type=head_type)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device); model.eval()
    
    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)
    cfg = AutoConfig.from_pretrained(args.hf_repo)
    fpc = getattr(cfg, "frames_per_clip", 16)

    # Info del modelo
    model_size_mb = get_model_size_mb(model)
    n_params = sum(p.numel() for p in model.parameters())

    path_solution = Path(args.solution_csv)
    if not path_solution.exists(): sys.exit(f"[ERR] CSV not found: {path_solution}")
    
    all_rows = read_solution_csv(path_solution)
    public_rows = [r for r in all_rows if "public" in r["usage"]]
    private_rows = [r for r in all_rows if "private" in r["usage"]]

    splits = []
    # Modo normal (solution.csv con Usage)
    if args.split in ["public", "both"] and len(public_rows) > 0:
        splits.append(("public", public_rows))
    if args.split in ["private", "both"] and len(private_rows) > 0:
        splits.append(("private", private_rows))

    # Fallback: CSV sin Usage (p.ej. val.csv de extracción) → un solo split "val"
    if not splits:
        splits.append(("val", all_rows))


    print("\n" + "="*60)
    print(f"MODELO: {head_type.upper()} | Tag: {run_tag}")
    print(f"STATS: Params={n_params/1e6:.1f}M | Size={model_size_mb:.1f} MB")
    print(f"CONFIG: Stride={args.stride_s}s | Agg={args.agg} | TTA-Thr={args.tta_thr}")
    print("="*60)

    for name, rows in splits:
        vid_dir = Path(args.video_root)
        if (vid_dir / f"test-{name}").exists(): vid_dir = vid_dir / f"test-{name}"
        
        res = evaluate_split(name, rows, model, processor, fpc, args.hist_s, args.stride_s,
                             device, vid_dir, args.agg, args.top_k, args.batch_size_infer, args.tta_thr, run_tag)
        
        if not res: continue
        print(f"\n[{name.upper()}]")
        print(f"  Videos evaluados     : {res['eval']} (Missing: {res['miss']})")
        print(f"  AP_anytime           : {res['ap']:.6f}")
        print(f"  AUC_anytime          : {res['auc']:.6f}")
        print(f"  Acc@0.5              : {res['acc']*100:.2f}%")
        print(f"  F1@0.5               : {res['f1']:.3f} (P:{res['prec']:.2f} R:{res['rec']:.2f})")
        print(f"  Confusion [TN FP/FN TP]: {res['cm']}")
        print(f"  mTTA@{res['mtta_thr']:.2f}           : {res['mtta']:.3f} s")
        print(f"  Recall_Before_Event  : {res['recall_before_event']*100:.2f}%")
        print(f"  Speed                : {res['infer_ms']:.1f} ms/vid ({res['xrt']:.1f}x real-time)")
        print(f"thr_R80             : {res.get('thr_r80', float('nan')):.3f}")
        print(f"Acc@R80             : {res.get('acc_r80', 0.0)*100:.2f}%")
        print(f"F1@R80              : {res.get('f1_r80', 0.0):.3f} (P:{res.get('prec_r80', 0.0):.2f} R:{res.get('rec_r80', 0.0):.2f})")
        print(f"mTTA@R80            : {res.get('mtta_r80', float('nan')):.3f} s")
        print(f"Rec_before_event@R80: {res.get('recall_before_event_r80', 0.0)*100:.2f}%")


if __name__ == "__main__":
    main()
