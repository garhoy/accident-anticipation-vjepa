from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def float_or_none(x: Any) -> Optional[float]:
    """
    Convierte a float o devuelve None.
    Soporta strings vacíos y separador decimal ','.
    """
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def get_balanced_subset(
    csv_path: Path,
    n_total: int = 60,
    *,
    target_col: str = "target",
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict]:
    """
    Lee un CSV y devuelve un subset balanceado (50% pos / 50% neg) en memoria.

    Requisitos del CSV:
      - columna `target_col` con valores {0,1} (acepta '0', '1', '0.0', '1.0')
    """
    if verbose:
        print(f"[DEBUG] Leyendo subset balanceado desde: {csv_path}")

    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pos_rows: List[Dict] = []
    neg_rows: List[Dict] = []

    for r in rows:
        try:
            t = int(float(str(r.get(target_col, "")).strip()))
        except Exception:
            continue
        if t == 1:
            pos_rows.append(r)
        elif t == 0:
            neg_rows.append(r)

    rng = random.Random(seed)
    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)

    n_target = max(0, int(n_total) // 2)
    actual_pos = min(len(pos_rows), n_target)
    actual_neg = min(len(neg_rows), n_target)

    subset = pos_rows[:actual_pos] + neg_rows[:actual_neg]
    rng.shuffle(subset)

    if verbose:
        print(f"[DEBUG] Subset: {len(subset)} ({actual_pos} Pos / {actual_neg} Neg)")
    return subset
