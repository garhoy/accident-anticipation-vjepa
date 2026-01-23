# metrics.py
# -*- coding: utf-8 -*-
"""
Métricas para detección/anticipación de accidentes.

Por defecto reproduce el protocolo de CRASH/LATTE:
- AP/AUC a nivel VÍDEO con score ANYTIME:
    s_video = max_t max_h p(t, h)   (sin restricción a pre-evento)
- Umbrales globales (R=0.8, maxF1) calculados sobre ese score ANYTIME.
- mTTA (threshold-free) y TTA@R80 calculados con PRIMER CRUCE PRE-EVENTO.

También se devuelven métricas "PRE" (pre-evento) para comparación justa
(online/causal), pero NO se usan para AP/umbrales por defecto.

Convenciones de tensores:
- logits_bt_w / probs_bt_w: (B, T, nW)
- mask_bt: (B, T) bool (frames válidos)
- deltas_bt: (B, T) "FIN-ANCLA (s)" según tu pipeline (se usa con dt_shift)
- horizons (nW): lista de ventanas de anticipación en segundos (p.ej., [1,2,3,4])
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------- Umbrales (q-grid) -----------------
def get_q_grid(grid: str | np.ndarray = "paper19") -> np.ndarray:
    """
    Grids típicos para mTTA:
      - 'classic9'  -> 0.1..0.9 (9 pts)
      - 'paper19'   -> 0.05..0.95 (19 pts) [muy usado]
      - 'dense99'   -> 0.01..0.99 (99 pts)
      - np.ndarray  -> custom
    """
    if isinstance(grid, np.ndarray):
        return grid.astype(np.float64)
    g = str(grid).lower()
    if g == "classic9":
        return np.arange(0.1, 1.0, 0.1, dtype=np.float64)
    if g == "dense99":
        return np.linspace(0.01, 0.99, 99, dtype=np.float64)
    return np.linspace(0.05, 0.95, 19, dtype=np.float64)


# ---------- Aggregación frame -> vídeo (PRE-EVENTO, por horizonte) ----------
@torch.no_grad()
def video_probs_from_frame_probs(
    probs_bt_w: torch.Tensor,   # (B,T,nW) en [0,1]
    mask_bt: torch.Tensor,      # (B,T)    bool
    deltas_bt: torch.Tensor,    # (B,T)    FIN-ANCLA (s)
    horizons: List[float],
    dt_shift: float = 0.0,      # corrige FIN->(start/center/end)
) -> np.ndarray:
    """
    Vídeo score por horizonte = max de prob en frames con (delta_eff <= -h),
    donde delta_eff = deltas_bt - dt_shift. (Vista PRE-EVENTO).
    Devuelve array (B, nW).
    """
    B, T, nW = probs_bt_w.shape
    out = np.zeros((B, nW), dtype=np.float32)
    delta_eff = deltas_bt - float(dt_shift)
    for j, h in enumerate(horizons):
        valid = (delta_eff <= -float(h)) & mask_bt
        masked = torch.where(valid, probs_bt_w[:, :, j], torch.zeros_like(probs_bt_w[:, :, j]))
        out[:, j] = masked.max(dim=1).values.cpu().numpy()
    return out


# ---------- Aggregación frame -> vídeo (ANYTIME, por horizonte) ----------
@torch.no_grad()
def video_probs_anytime_from_frame_probs(
    probs_bt_w: torch.Tensor,   # (B,T,nW) en [0,1]
    mask_bt: torch.Tensor,      # (B,T)    bool
) -> np.ndarray:
    """
    Vídeo score por horizonte SIN restringir a pre-evento:
      s_video[j] = max_t p(t, j) sobre frames válidos.
    Devuelve (B, nW).
    """
    B, T, nW = probs_bt_w.shape
    out = np.zeros((B, nW), dtype=np.float32)
    for j in range(nW):
        masked = torch.where(mask_bt, probs_bt_w[:, :, j], torch.zeros_like(probs_bt_w[:, :, j]))
        out[:, j] = masked.max(dim=1).values.cpu().numpy()
    return out


# ---------- AP/AUC a nivel vídeo (por horizonte y promedio simple) ----------
def ap_auc_video(video_probs: np.ndarray, y_video: np.ndarray, horizons: List[float]) -> Tuple[float, float, Dict[str, float]]:
    per_w: Dict[str, float] = {}
    apv, aucv = [], []
    yv = (y_video > 0.5).astype(np.float32)
    for j, h in enumerate(horizons):
        # AP (área bajo PR). Si no hay positivos, AP=0 por convención.
        ap = average_precision_score(yv, video_probs[:, j]) if yv.sum() > 0 else 0.0
        # AUC (ROC). Si y es constante, AUC=0.5 neutro.
        auc = roc_auc_score(yv, video_probs[:, j]) if (yv.min() != yv.max()) else 0.5
        per_w[f"{h}s_ap"] = float(ap)
        per_w[f"{h}s_auc"] = float(auc)
        apv.append(ap); aucv.append(auc)
    return float(np.mean(apv)), float(np.mean(aucv)), per_w


# ===================== VISTA GLOBAL "PRE" (pre-evento) =====================
@torch.no_grad()
def video_scores_alltime(
    probs_bt_w: torch.Tensor,   # (B,T,nW)
    mask_bt: torch.Tensor,      # (B,T)
    deltas_bt: torch.Tensor,    # (B,T)
    dt_shift: float = 0.0,
) -> np.ndarray:
    """
    Score único de vídeo colapsando horizontes y tiempo, SOLO pre-evento:
      - max sobre horizontes -> p_bt = max_h p(B,T,h)
      - max antes del evento (delta_eff <= 0) -> max_t p_bt
    Devuelve (B,).
    """
    probs_bt = probs_bt_w.max(dim=2).values               # (B,T)
    delta_eff = deltas_bt - float(dt_shift)
    valid    = (delta_eff <= 0) & mask_bt                 # (B,T)
    masked   = torch.where(valid, probs_bt, torch.zeros_like(probs_bt))
    return masked.max(dim=1).values.cpu().numpy()         # (B,)


# ===================== VISTA GLOBAL "ANYTIME" (CRASH/LATTE) =================
@torch.no_grad()
def video_scores_anytime(
    probs_bt_w: torch.Tensor,   # (B,T,nW)
    mask_bt: torch.Tensor,      # (B,T)
) -> np.ndarray:
    """
    Score único de vídeo colapsando horizontes y tiempo SIN restricción temporal:
      s_video = max_t max_h p(t,h)  sobre frames válidos.
    Devuelve (B,).
    """
    probs_bt = probs_bt_w.max(dim=2).values               # (B,T)
    masked   = torch.where(mask_bt, probs_bt, torch.zeros_like(probs_bt))
    return masked.max(dim=1).values.cpu().numpy()         # (B,)


# ---------- AP/AUC con score global (independiente de horizontes) ----------
def ap_auc_video_lit(video_scores: np.ndarray, y_video: np.ndarray) -> Tuple[float, float]:
    yv = (y_video > 0.5).astype(np.int32)
    ap  = average_precision_score(yv, video_scores) if yv.sum() > 0 else 0.0
    auc = roc_auc_score(yv, video_scores) if (yv.min() != yv.max()) else 0.5
    return float(ap), float(auc)


# ---- Umbrales globales (estilo CCD/CRASH/LATTE) ----
def threshold_from_val_recall(video_scores: np.ndarray, y_video: np.ndarray,
                              R: float = 0.8,
                              q_grid: np.ndarray = np.linspace(0.01, 0.99, 99)) -> float:
    """
    Umbral mínimo q tal que recall >= R sobre positivos, en un grid.
    Evita q=0.0 para no devolver 0 por construcción.
    """
    ppos = video_scores[(y_video > 0.5)]
    if ppos.size == 0:
        return 1.0
    for q in q_grid:
        if (ppos >= q).mean() >= R:
            return float(q)
    return float(q_grid[-1])


def threshold_from_val_max_f1(video_scores: np.ndarray, y_video: np.ndarray) -> float:
    y = (y_video > 0.5).astype(int)
    q_grid = np.linspace(0.01, 0.99, 99)
    best_q, best_f1 = 0.5, -1.0
    for q in q_grid:
        yhat = (video_scores >= q).astype(int)
        tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
        fn = int(((y == 1) & (yhat == 0)).sum())
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_q = f1, float(q)
    return best_q


# ---------- Paquete de métricas para VALIDACIÓN ----------
@torch.no_grad()
def compute_val_metrics_from_frame_logits(
    logits_bt_w: torch.Tensor,  # (B,T,nW) logits
    mask_bt: torch.Tensor,      # (B,T)
    deltas_bt: torch.Tensor,    # (B,T) FIN-ANCLA (s)
    y_video_t: torch.Tensor,    # (B,)
    horizons: List[float],
    dt_shift: float = 0.0,      # corrige al punto de decisión (solo afecta a PRE)
    ap_scope: str = "anytime",  # {"anytime","pre"} -> por defecto ANYTIME (CRASH/LATTE)
    times_bt=None, t_events=None,
) -> Dict:
    """
    Devuelve métricas a nivel vídeo.
    - "ANYTIME" (por defecto): AP/AUC y umbrales con max en TODO el clip (igual a CRASH/LATTE).
    - "PRE"                 : versión estricta usando sólo pre-evento (tu vista original).
    Además, devuelve ambas vistas en 'extras' para comparación.
    """
    probs_bt_w = torch.sigmoid(logits_bt_w)
    y_video = y_video_t.cpu().numpy().astype(int)


    # --- PRE usando evento (si viene) ---
    if times_bt is not None and t_events is not None:
        # t_ev = torch.tensor(t_events, device=times_bt.device, dtype=times_bt.dtype).unsqueeze(1)  # (B,1)
        t_ev = torch.as_tensor(t_events, device=times_bt.device, dtype=times_bt.dtype).unsqueeze(1)  # (B,1)
        valid_pre = (times_bt <= t_ev) & mask_bt
        # per-window PRE
        video_probs_pre = []
        for j,_ in enumerate(horizons):
            pj = probs_bt_w[:, :, j]
            masked = torch.where(valid_pre, pj, torch.zeros_like(pj))
            video_probs_pre.append(masked.max(dim=1).values.cpu().numpy())
        video_probs_pre = np.stack(video_probs_pre, axis=1)
        # global PRE
        p_bt = probs_bt_w.max(dim=2).values
        masked = torch.where(valid_pre, p_bt, torch.zeros_like(p_bt))
        video_scores_pre = masked.max(dim=1).values.cpu().numpy()
    else:
        # fallback (ancla FIN como antes)
        video_probs_pre = video_probs_from_frame_probs(probs_bt_w, mask_bt, deltas_bt, horizons, dt_shift)
        video_scores_pre = video_scores_alltime(probs_bt_w, mask_bt, deltas_bt, dt_shift)

    # -------- Vista PRE (pre-evento) --------
    
    # video_probs_pre = video_probs_from_frame_probs(
    #     probs_bt_w, mask_bt, deltas_bt, horizons, dt_shift=dt_shift
    # )
    
    ap_video_pre, auc_video_pre, per_window_pre = ap_auc_video(video_probs_pre, y_video, horizons)
    # video_scores_pre = video_scores_alltime(probs_bt_w, mask_bt, deltas_bt, dt_shift=dt_shift)
    ap_lit_pre, auc_lit_pre = ap_auc_video_lit(video_scores_pre, y_video)
    thr_R80_pre   = threshold_from_val_recall(video_scores_pre, y_video, R=0.8)
    thr_maxF1_pre = threshold_from_val_max_f1(video_scores_pre, y_video)
    
    # -------- Vista ANYTIME (CRASH/LATTE) --------
    video_probs_any = video_probs_anytime_from_frame_probs(probs_bt_w, mask_bt)
    ap_video_any, auc_video_any, per_window_any = ap_auc_video(video_probs_any, y_video, horizons)
    video_scores_any = video_scores_anytime(probs_bt_w, mask_bt)
    ap_lit_any, auc_lit_any = ap_auc_video_lit(video_scores_any, y_video)
    thr_R80_any   = threshold_from_val_recall(video_scores_any, y_video, R=0.8)
    thr_maxF1_any = threshold_from_val_max_f1(video_scores_any, y_video)

    # -------- Selección de la VISTA “oficial” (claves canónicas) --------
    use_any = (str(ap_scope).lower() == "anytime")
    AP_video      = ap_video_any   if use_any else ap_video_pre
    AUC_video     = auc_video_any  if use_any else auc_video_pre
    per_window    = per_window_any if use_any else per_window_pre
    AP_video_lit  = ap_lit_any     if use_any else ap_lit_pre
    AUC_video_lit = auc_lit_any    if use_any else auc_lit_pre
    thr_R80_lit   = thr_R80_any    if use_any else thr_R80_pre
    thr_maxF1_lit = thr_maxF1_any  if use_any else thr_maxF1_pre

    return {
        # —— claves canónicas (por defecto, ANYTIME = CRASH/LATTE) ——
        "AP_video": AP_video,
        "AUC_video": AUC_video,
        "per_window": per_window,
        "AP_video_lit": AP_video_lit,
        "AUC_video_lit": AUC_video_lit,
        "thr_R80_lit": float(thr_R80_lit),
        "thr_maxF1_lit": float(thr_maxF1_lit),

        # —— extras para diagnóstico/ablation ——
        "extras": {
            "anytime": {
                "AP_video": ap_video_any, "AUC_video": auc_video_any,
                "per_window": per_window_any,
                "AP_video_lit": ap_lit_any, "AUC_video_lit": auc_lit_any,
                "thr_R80_lit": float(thr_R80_any), "thr_maxF1_lit": float(thr_maxF1_any),
            },
            "pre": {
                "AP_video": ap_video_pre, "AUC_video": auc_video_pre,
                "per_window": per_window_pre,
                "AP_video_lit": ap_lit_pre, "AUC_video_lit": auc_lit_pre,
                "thr_R80_lit": float(thr_R80_pre), "thr_maxF1_lit": float(thr_maxF1_pre),
            },
        }
    }


# ---- mTTA (threshold-free, PRE-EVENTO) ----
@torch.no_grad()
def mtta_threshold_free_lit(
    probs_bt_w: torch.Tensor,   # (B,T,nW) en [0,1]
    times_bt: torch.Tensor,     # (B,T)  --IGNORADO, se mantiene por compatibilidad
    mask_bt: torch.Tensor,      # (B,T)
    t_events: np.ndarray,       # (B,)   --IGNORADO, se mantiene por compatibilidad
    y_video: np.ndarray,        # (B,)
    q_grid: np.ndarray | str = "paper19",
    strict_no_cross: bool = True,
    *,
    deltas_bt: torch.Tensor,    # (B,T)  FIN-ANCLA (s)
    dt_shift: float = 0.0,      # corrige anchor al punto de decisión
) -> float:
    """
    mTTA threshold-free usando EXCLUSIVAMENTE deltas_bt (FIN-ANCLA en segundos).
    Para cada q:
      - PRE-EVENTO: delta_eff <= 0
      - primer cruce p(t) >= q
      - TTA = max(0, -delta_eff[t_cross])
    """
    qs = get_q_grid(q_grid)
    p_bt = probs_bt_w.max(dim=2).values            # (B,T)
    delta_eff = deltas_bt - float(dt_shift)        # (B,T)
    mttas: list[float] = []

    for q in qs:
        ttas_q: list[float] = []
        for i in range(p_bt.size(0)):
            if y_video[i] < 0.5:
                continue
            m = mask_bt[i]
            p = p_bt[i][m]                         # (Ti,)
            d = delta_eff[i][m]                    # (Ti,)
            pre = d <= 0.0
            if not pre.any():
                if strict_no_cross:
                    ttas_q.append(0.0)
                continue
            p_pre = p[pre]; d_pre = d[pre]
            idx = (p_pre >= q).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                if strict_no_cross:
                    ttas_q.append(0.0)
                continue
            # tiempo a evento en el primer cruce
            tta = float((-d_pre[idx[0]]).item())
            ttas_q.append(max(0.0, tta))
        if ttas_q:
            mttas.append(float(np.mean(ttas_q)))
    return float(np.mean(mttas)) if mttas else 0.0


@torch.no_grad()
def tta_at_threshold(
    probs_bt_w: torch.Tensor,   # (B,T,nW)
    times_bt: torch.Tensor,     # (B,T)  --IGNORADO, se mantiene por compatibilidad
    mask_bt: torch.Tensor,      # (B,T)
    t_events: np.ndarray,       # (B,)   --IGNORADO, se mantiene por compatibilidad
    thr: float,                 # umbral global (p.ej., thr@R80 ANYTIME)
    y_video: np.ndarray,        # (B,)
    strict_no_cross: bool = True,
    *,
    deltas_bt: torch.Tensor,    # (B,T) FIN-ANCLA (s)
    dt_shift: float = 0.0,
) -> float:
    """
    TTA a umbral fijo usando deltas_bt:
      - PRE-EVENTO: delta_eff <= 0
      - primer cruce p(t) >= thr
      - TTA = max(0, -delta_eff[t_cross])
    """
    p_bt = probs_bt_w.max(dim=2).values
    delta_eff = deltas_bt - float(dt_shift)
    ttas: list[float] = []
    for i in range(p_bt.size(0)):
        if y_video[i] < 0.5:
            continue
        m = mask_bt[i]
        p = p_bt[i][m]
        d = delta_eff[i][m]
        pre = d <= 0.0
        if not pre.any():
            if strict_no_cross:
                ttas.append(0.0)
            continue
        p_pre = p[pre]; d_pre = d[pre]
        idx = (p_pre >= thr).nonzero(as_tuple=True)[0]
        if idx.numel():
            tta = float((-d_pre[idx[0]]).item())
            ttas.append(max(0.0, tta))
        else:
            if strict_no_cross:
                ttas.append(0.0)
    return float(np.mean(ttas)) if ttas else 0.0