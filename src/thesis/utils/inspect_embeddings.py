#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspección de embeddings .pt (V-JEPA):
- Recorre recursivamente directorios dados y carga .pt.
- Extrae: target (meta), target inferido por carpeta, T,D, fps, clip_len,
  duration, time_of_event, window, NaNs, etc.
- Reporta resumen por consola y guarda un CSV.

Uso:
  poetry run python scripts/inspect_embeddings.py \
    --roots data_exp/vjepa2_l16s1_f20_hist5/dad/train data_exp/vjepa2_l16s1_f20_hist5/dad/val \
    --out inspect_embeddings_dad.csv
"""

from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

POS_NAMES = {"pos","positive","positives","crash","crashes","accident","accidents","c"}
NEG_NAMES = {"neg","negative","negatives","normal","nonaccident","n"}

def infer_target_from_path(p: Path) -> Optional[int]:
    parent = p.parent.name.lower()
    if any(k in parent for k in POS_NAMES): return 1
    if any(k in parent for k in NEG_NAMES): return 0
    return None

def safe_get(d: Dict[str,Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def describe_one(pt_path: Path) -> Optional[Dict[str,Any]]:
    try:
        data = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        return {"path": str(pt_path), "error": f"load_error: {e}"}

    meta: Dict[str,Any] = data.get("meta", {}) or {}
    z = data.get("z_clip", None)
    T, D = (int(z.shape[0]), int(z.shape[1])) if isinstance(z, torch.Tensor) and z.ndim==2 else (0, 0)
    has_nan = bool(isinstance(z, torch.Tensor) and torch.isnan(z).any())

    fps = float(data.get("fps_target", meta.get("fps_target", float("nan"))))
    clip_len = int(meta.get("clip_len", data.get("clip_len", 16)))
    duration = meta.get("duration", None)
    t_event = meta.get("time_of_event", None)
    window = meta.get("window", None)
    delta_key = "delta_to_anchor" if "delta_to_anchor" in data else None
    clip_times_key = "clip_times" if "clip_times" in data else None

    target_meta = meta.get("target", None)
    target_meta_i = int(target_meta) if target_meta is not None else None
    target_inferred = infer_target_from_path(pt_path)
    # “Efectivo” como lo usa tu dataset actual: prioriza meta.target si existe
    target_effective = target_meta_i if target_meta_i is not None else (target_inferred if target_inferred is not None else 0)

    mismatch = (target_meta_i is not None and target_inferred is not None and target_meta_i != target_inferred)

    return {
        "path": str(pt_path),
        "parent": pt_path.parent.name,
        "file": pt_path.name,
        "T": T, "D": D,
        "has_nan": has_nan,
        "fps_target": fps if not math.isnan(fps) else "",
        "clip_len": clip_len,
        "duration": duration if duration is not None else "",
        "time_of_event": t_event if t_event is not None else "",
        "window": str(tuple(window)) if isinstance(window,(list,tuple)) else "",
        "has_delta_to_anchor": delta_key is not None,
        "has_clip_times": clip_times_key is not None,
        "target_meta": target_meta_i if target_meta_i is not None else "",
        "target_inferred": target_inferred if target_inferred is not None else "",
        "target_effective": target_effective,
        "mismatch_meta_vs_folder": bool(mismatch),
        "error": "",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Directorios raíz a inspeccionar (se recorren recursivamente).")
    ap.add_argument("--out", default="inspect_embeddings.csv", help="Ruta de salida CSV.")
    ap.add_argument("--max", type=int, default=0, help="Procesar como máximo N ficheros (0 = todos).")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    files: List[Path] = []
    for r in roots:
        if not r.exists():
            print(f"[WARN] No existe: {r}", file=sys.stderr)
            continue
        files.extend(sorted(r.rglob("*.pt")))
    if args.max > 0:
        files = files[:args.max]

    rows: List[Dict[str,Any]] = []
    for p in files:
        rec = describe_one(p)
        if rec is None: 
            continue
        rows.append(rec)

    # Resumen
    total = len(rows)
    load_errors = sum(1 for r in rows if r.get("error"))
    valid = sum(1 for r in rows if not r.get("error") and r.get("T",0) > 0 and r.get("D",0) > 0)
    with_meta_target = sum(1 for r in rows if r.get("target_meta") != "")
    with_inferred = sum(1 for r in rows if r.get("target_inferred") != "")
    mismatches = sum(1 for r in rows if r.get("mismatch_meta_vs_folder", False))
    pos_eff = sum(1 for r in rows if r.get("target_effective",0) == 1)
    neg_eff = sum(1 for r in rows if r.get("target_effective",0) == 0)

    print("\n===== INSPECTION SUMMARY =====")
    print(f"Files total             : {total}")
    print(f"Loaded OK (T>0,D>0)     : {valid}")
    print(f"Load errors             : {load_errors}")
    print(f"meta.target present     : {with_meta_target}")
    print(f"target inferred by path : {with_inferred}")
    print(f"meta vs path mismatches : {mismatches}")
    print(f"Effective targets -> pos: {pos_eff} | neg: {neg_eff}")

    # Stats básicas de T y D
    Ts = [int(r["T"]) for r in rows if not r.get("error") and r.get("T",0)>0]
    Ds = [int(r["D"]) for r in rows if not r.get("error") and r.get("D",0)>0]
    if Ts:
        import numpy as np
        print(f"T length: mean={np.mean(Ts):.1f}  min={np.min(Ts)}  p50={np.percentile(Ts,50):.1f}  p90={np.percentile(Ts,90):.1f}  max={np.max(Ts)}")
    if Ds:
        from collections import Counter
        print("D dims  :", dict(Counter(Ds)))

    # Guardar CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path","parent","file","T","D","has_nan","fps_target","clip_len",
        "duration","time_of_event","window","has_delta_to_anchor","has_clip_times",
        "target_meta","target_inferred","target_effective","mismatch_meta_vs_folder","error"
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV guardado en: {out}")

    # Muestra 3 ejemplos por clase efectiva
    def show_examples(label: int, k: int = 3):
        print(f"\n--- Ejemplos target_effective={label} ---")
        count = 0
        for r in rows:
            if r.get("target_effective",0) == label and not r.get("error"):
                print(f"* {r['path']} | T={r['T']} D={r['D']} meta.target={r.get('target_meta','')} "
                      f"inferred={r.get('target_inferred','')} evt={r.get('time_of_event','')}")
                count += 1
                if count >= k: break
        if count == 0:
            print("(sin ejemplos)")

    show_examples(1, 3)
    show_examples(0, 3)

if __name__ == "__main__":
    main()
