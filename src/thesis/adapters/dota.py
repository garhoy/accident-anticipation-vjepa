#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoTA adapter con YAML (estructura PLANA de frames):
  frames_root/<video_id>/images/*.jpg

Lee:
  annotations_dir/{metadata_train.json, metadata_val.json}  (val = test oficial)

Opcional:
  per_video_annotations_dir/<video_id>.json  (para filtrar ignore/ego/night, etc.)

Genera:
  meta_out_dir/ccd_meta.csv
  meta_out_dir/extractor_csv/{train.csv, val.csv, test.csv}

Negativos = precursor ('id__pre', 'id__pre0'..), sólo si hay suficiente contexto normal
(según min_neg_ctx_s y margen; o siempre si always_create_pre=true).

Uso:
  poetry run python adapters/dota.py configs/dota.yaml
"""
from __future__ import annotations
import argparse, csv, json, sys, os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import yaml  # pyyaml
except Exception:
    print("[ERROR] Falta 'pyyaml'. Instala: poetry add pyyaml (o pip install pyyaml)", file=sys.stderr)
    sys.exit(1)

# ----------------- utilidades -----------------
def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")

def load_json_any(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text())
    # Soporta dict {id: meta} o lista de objetos con "video_name" / "id"
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out = {}
        for it in data:
            vid = it.get("video_name") or it.get("id")
            if not vid:
                raise ValueError(f"Elemento sin 'video_name' ni 'id' en {p.name}")
            out[vid] = it
        return out
    raise TypeError(f"Formato no soportado en {p}")

def stratified_split(items: List[Tuple[str,dict]], key="anomaly_class",
                     val_ratio: float = 0.2, seed: int = 123):
    import random
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for vid, meta in items:
        buckets[meta.get(key, "UNK")].append((vid, meta))
    tr, va = [], []
    for _, lst in buckets.items():
        rng.shuffle(lst)
        n_va = max(1, int(round(len(lst) * val_ratio)))
        va.extend(lst[:n_va]); tr.extend(lst[n_va:])
    return tr, va

def load_per_video_meta(per_dir: Path, vid: str) -> dict | None:
    p = per_dir / f"{vid}.json"
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except Exception: return None

def write_ccd_meta(out_csv: Path, entries: List[Tuple[str,str,dict]], fps: float,
                   per_dir: Path | None, frames_root: Path, frames_subdir: str):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id","subset","anomaly_class","video_start","video_end",
            "anomaly_start","anomaly_end","num_frames",
            "anomaly_start_s","anomaly_end_s","duration_s",
            "ignore","ego_involve","night","channel","accident_name",
            "frames_dir_exists"
        ])
        for vid, subset, m in entries:
            a0 = int(m.get("anomaly_start", 0))
            a1 = int(m.get("anomaly_end",   a0))
            nf = int(m.get("num_frames",    max(a1+1, 1)))
            duration_s = nf / max(float(fps), 1e-6)

            pv = load_per_video_meta(per_dir, vid) if per_dir else {}
            ignore = pv.get("ignore", "")
            ego    = pv.get("ego_involve", "")
            night  = pv.get("night", "")
            channel= pv.get("channel", "")
            acc    = pv.get("accident_name", "") or pv.get("anomaly_class","")

            frames_dir = frames_root / vid / frames_subdir
            frames_ok = frames_dir.exists()

            w.writerow([
                vid, subset,
                m.get("anomaly_class",""),
                int(m.get("video_start",0)),
                int(m.get("video_end",  0)),
                a0, a1, nf,
                a0/float(fps), a1/float(fps), duration_s,
                ignore, ego, night, channel, acc,
                int(frames_ok)
            ])

def write_extractor_csv(out_csv: Path, rows: List[Tuple[str,int,float|None,float|None]]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","target","time_of_event","time_of_alert"])
        for vid, tgt, tev, tal in rows:
            w.writerow([
                vid, tgt,
                "" if tev is None else f"{tev:.3f}",
                "" if tal is None else f"{tal:.3f}"
            ])

def summarize_by_class(items: List[Tuple[str,dict]]) -> Dict[str,int]:
    cnt = defaultdict(int)
    for _, m in items:
        cnt[m.get("anomaly_class","UNK")] += 1
    return dict(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))

# ----------------- construcción de filas (con negativos) -----------------
def build_rows_with_precursor_negs(items: List[Tuple[str,dict]],
                                   fps: float, hist_s: float, margin: float,
                                   neg_window_s: float, k_negs: int,
                                   min_neg_ctx_s: float, always_create_pre: bool) -> List[Tuple[str,int,float|None,float|None]]:
    """
    Devuelve filas: (id, target, time_of_event, time_of_alert)
    - Positivo: id,1,anomaly_start/fps,""
    - Negativo (precursor): id__pre[, id__pre1..],0,"","" si hay contexto suficiente
      para ubicar una ventana negativa que NO alcance el evento.
    """
    rows = []
    for vid, m in items:
        a0 = int(m.get("anomaly_start", 0))
        nf = int(m.get("num_frames", max(a0+1,1)))
        tev = a0 / float(fps)
        duration_s = nf / max(float(fps), 1e-6)

        # 1) Positivo
        rows.append((vid, 1, tev, None))

        # 2) Negativos del precursor
        #    Consideramos viable si existe un t_end tal que:
        #    t_end >= min_neg_ctx_s y t_end <= min(tev - margin, neg_window_s, duration_s)
        t_end_min = float(min_neg_ctx_s)
        t_end_max = min(tev - float(margin), float(neg_window_s), float(duration_s))
        can_make = (t_end_max >= t_end_min + 1e-6)

        if k_negs > 0 and (can_make or always_create_pre):
            if k_negs == 1:
                rows.append((f"{vid}__pre", 0, None, None))
            else:
                for k in range(k_negs):
                    rows.append((f"{vid}__pre{k}", 0, None, None))
        # Si no hay espacio y always_create_pre=false, omitimos el negativo
    return rows

# ----------------- filtros por per-video annotations -----------------
def build_filtered_items(meta: Dict[str,dict],
                         per_dir: Path | None,
                         filter_ignore: bool,
                         require_ego: bool,
                         exclude_night: bool) -> List[Tuple[str,dict]]:
    out = []
    for vid, m in meta.items():
        if per_dir is None:
            out.append((vid, m)); continue
        pv = load_per_video_meta(per_dir, vid) or {}
        if filter_ignore and pv.get("ignore", False):
            continue
        if require_ego and not pv.get("ego_involve", False):
            continue
        if exclude_night and pv.get("night", False):
            continue
        out.append((vid, m))
    return out

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Ruta al YAML (configs/dota.yaml)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Rutas
    annotations_dir = Path(cfg["annotations_dir"])
    frames_root     = Path(cfg["frames_root"])
    meta_out_dir    = Path(cfg["meta_out_dir"])
    ensure_dir(annotations_dir); ensure_dir(frames_root)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    # Parametría
    fps          = float(cfg.get("fps", 10.0))
    hist_s       = float(cfg.get("hist_s", 5.0))
    margin       = float(cfg.get("margin", 0.3))
    neg_window_s = float(cfg.get("neg_window_s", 13.0))
    k_negs       = int(cfg.get("k_negs", 1))
    val_ratio    = float(cfg.get("val_ratio", 0.2))
    seed         = int(cfg.get("seed", 123))
    min_neg_ctx_s    = float(cfg.get("min_neg_ctx_s", 1.6))
    always_create_pre = bool(cfg.get("always_create_pre", False))

    use_pv   = bool(cfg.get("use_per_video_annotations", False))
    per_dir  = Path(cfg.get("per_video_annotations_dir","")) if use_pv else None
    if use_pv:
        ensure_dir(per_dir)

    filter_ignore = bool(cfg.get("filter_ignore", False))
    require_ego   = bool(cfg.get("require_ego_involve", False))
    exclude_night = bool(cfg.get("exclude_night", False))

    validate_frames    = bool(cfg.get("validate_frames", False))
    frames_subdir_name = str(cfg.get("frames_subdir_name", "images"))

    # Cargar metadatos (train y test oficial)
    mt = annotations_dir / "metadata_train.json"
    mv = annotations_dir / "metadata_val.json"  # 'val' = test oficial
    meta_tr = load_json_any(mt)
    meta_te = load_json_any(mv)

    # Filtrar por per-video annotations si se pide
    items_tr_all = build_filtered_items(meta_tr, per_dir if use_pv else None,
                                        filter_ignore, require_ego, exclude_night)
    items_te     = build_filtered_items(meta_te, per_dir if use_pv else None,
                                        filter_ignore, require_ego, exclude_night)

    # Split interno val desde el train oficial (estratificado por anomaly_class)
    items_tr, items_va = stratified_split(items_tr_all, key="anomaly_class",
                                          val_ratio=val_ratio, seed=seed)

    # Validación de frames (opcional: todos; por defecto, omite para ir rápido)
    if validate_frames:
        miss = []
        for subset, items in (("train",items_tr), ("val",items_va), ("test",items_te)):
            for vid, _ in items:
                p = frames_root / vid / frames_subdir_name
                if not p.exists():
                    miss.append((subset, vid, str(p)))
        if miss:
            print(f"[WARN] {len(miss)} ids sin frames_dir:", file=sys.stderr)
            for s, vid, p in miss[:10]:
                print(f"  - {s} {vid} -> {p}", file=sys.stderr)

    # Auditoría (ccd_meta.csv)
    entries = ([(vid,"train",m) for vid,m in items_tr] +
               [(vid,"val",  m) for vid,m in items_va] +
               [(vid,"test", m) for vid,m in items_te])
    write_ccd_meta(meta_out_dir / "ccd_meta.csv", entries, fps=fps,
                   per_dir=(per_dir if use_pv else None),
                   frames_root=frames_root, frames_subdir=frames_subdir_name)

    # CSVs del extractor (positivos + negativos del precursor)
    rows_tr = build_rows_with_precursor_negs(items_tr, fps, hist_s, margin,
                                             neg_window_s, k_negs,
                                             min_neg_ctx_s, always_create_pre)
    rows_va = build_rows_with_precursor_negs(items_va, fps, hist_s, margin,
                                             neg_window_s, k_negs,
                                             min_neg_ctx_s, always_create_pre)
    rows_te = build_rows_with_precursor_negs(items_te, fps, hist_s, margin,
                                             neg_window_s, 0,    # test sin negativos
                                             min_neg_ctx_s, always_create_pre=False)

    ext_dir = meta_out_dir / "extractor_csv"
    write_extractor_csv(ext_dir / "train.csv", rows_tr)
    write_extractor_csv(ext_dir / "val.csv",   rows_va)
    write_extractor_csv(ext_dir / "test.csv",  rows_te)

    # Resumen
    pos_tr = sum(1 for r in rows_tr if r[1]==1); neg_tr = sum(1 for r in rows_tr if r[1]==0)
    pos_va = sum(1 for r in rows_va if r[1]==1); neg_va = sum(1 for r in rows_va if r[1]==0)
    pos_te = sum(1 for r in rows_te if r[1]==1); neg_te = sum(1 for r in rows_te if r[1]==0)
    print("[OK] ccd_meta.csv y extractor_csv/{train,val,test}.csv generados en:", ext_dir.parent)
    print(f"[INFO] Train: +{pos_tr} / -{neg_tr} | Val: +{pos_va} / -{neg_va} | Test: +{pos_te} / -{neg_te}")

    # (Opcional) pequeños resúmenes por clase tras filtros + split
    tr_cls = summarize_by_class(items_tr)
    va_cls = summarize_by_class(items_va)
    te_cls = summarize_by_class(items_te)
    def _fmt(d): return ", ".join([f"{k}:{v}" for k,v in list(d.items())[:10]]) + (" ..." if len(d)>10 else "")
    print(f"[CLS] top train: {_fmt(tr_cls)}")
    print(f"[CLS] top val:   {_fmt(va_cls)}")
    print(f"[CLS] top test:  {_fmt(te_cls)}")
    print(f"[HINT] Estructura frames esperada: {frames_root}/<id>/{frames_subdir_name}/*.jpg  (fps={fps})")

if __name__ == "__main__":
    main()
