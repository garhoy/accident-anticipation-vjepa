#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

# Paths are resolved relative to this file so it works regardless of cwd
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

# ==================== CONFIGURA TUS RUTAS AQUÍ ====================

# 1. Pega la ruta al CSV de scores que YA generaste
PATH_TO_PREDICTIONS = BASE_DIR / "predictions" / "nexar_predictions_20251030-001809.csv"

# 2. Asegúrate de que esta es la ruta a tu CSV CON etiquetas (target)
PATH_TO_GROUND_TRUTH = REPO_ROOT / "V-JEPA-2" / "data" / "metadata" / "Nexar" / "test.csv"

# ==================================================================

def _float_or_none(x: Any) -> Optional[float]:
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None

def load_scores(p: Path) -> Dict[str, float]:
    """Carga el CSV de predicciones (id, score, label)"""
    scores = {}
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "id" not in (r.fieldnames or []) or "score" not in (r.fieldnames or []):
            raise ValueError(f"El CSV de predicciones debe tener 'id' y 'score'. Encontradas: {r.fieldnames}")
        for row in r:
            score = _float_or_none(row.get("score"))
            if score is not None:
                scores[str(row["id"])] = score
    return scores

def load_ground_truth(p: Path) -> Dict[str, int]:
    """Carga el CSV de etiquetas (id, target)"""
    labels = {}
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "id" not in (r.fieldnames or []) or "target" not in (r.fieldnames or []):
            raise ValueError(f"El CSV de ground truth debe tener 'id' y 'target'. Encontradas: {r.fieldnames}")
        for row in r:
            target = _float_or_none(row.get("target"))
            if target is not None:
                labels[str(row["id"])] = 1 if target == 1.0 else 0
    return labels

# ============== Fallbacks NumPy AP/AUC (copiados de tu script) =================
def average_precision_np(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AP con integración por pasos (estilo sklearn, sin dependencias)."""
    y_true = (y_true > 0).astype(np.int32)
    order = np.argsort(-y_scores)
    y = y_true[order]
    P = int(np.sum(y))
    if P == 0:
        return 0.0
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / P
    idx_pos = np.where(y == 1)[0]
    if idx_pos.size == 0:
        return 0.0
    recall_prev = np.concatenate(([0.0], recall[idx_pos[:-1]]))
    ap = np.sum((recall[idx_pos] - recall_prev) * precision[idx_pos])
    return float(ap)

def roc_auc_np(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """ROC AUC por trapecios sobre TPR(FPR)."""
    y_true = (y_true.astype(int) > 0).astype(np.int32)
    order = np.argsort(-y_scores)
    y = y_true[order]
    P = int(np.sum(y == 1))
    N = int(np.sum(y == 0))
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / P
    fpr = fp / N
    
    return float(np.trapz(tpr, fpr))
# ==============================================================================

def main():
    print(f"[INFO] Cargando scores de: {PATH_TO_PREDICTIONS}")
    if not PATH_TO_PREDICTIONS.exists():
        print(f"[ERR] No se encuentra el archivo de predicciones. Revisa la ruta.")
        return
    scores_map = load_scores(PATH_TO_PREDICTIONS)
    print(f"[OK] {len(scores_map)} scores cargados.")
    
    print(f"[INFO] Cargando etiquetas de: {PATH_TO_GROUND_TRUTH}")
    if not PATH_TO_GROUND_TRUTH.exists():
        print(f"[ERR] No se encuentra el archivo de ground truth. Revisa la ruta.")
        return
    labels_map = load_ground_truth(PATH_TO_GROUND_TRUTH)
    print(f"[OK] {len(labels_map)} etiquetas cargadas.")

    y_scores_list = []
    y_true_list = []
    
    missing_in_preds = 0
    
    # Emparejar scores con etiquetas
    for vid_id, target in labels_map.items():
        score = scores_map.get(vid_id)
        
        if score is None:
            # Este vídeo estaba en el ground truth, pero no en tus predicciones
            # (quizás un error al procesar, o el vídeo faltaba)
            missing_in_preds += 1
            continue
            
        y_scores_list.append(score)
        y_true_list.append(target)

    if not y_scores_list:
        print("[ERR] No se pudo emparejar ningún score con ninguna etiqueta. ¿Son los mismos IDs?")
        return
        
    if missing_in_preds > 0:
        print(f"[WARN] {missing_in_preds} IDs del ground truth no tenían score. Se ignorarán para las métricas.")

    y_true = np.array(y_true_list, dtype=np.int32)
    y_scores = np.array(y_scores_list, dtype=np.float32)

    ap = average_precision_np(y_true, y_scores)
    auc = roc_auc_np(y_true, y_scores)

    print("\n" + "=" * 60)
    print("  MODO: EVALUACIÓN (desde archivos CSV)")
    print(f"  Vídeos evaluados: {len(y_scores)}")
    print(f"  AP:  {ap:.6f}")
    print(f"  AUC: {auc:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
