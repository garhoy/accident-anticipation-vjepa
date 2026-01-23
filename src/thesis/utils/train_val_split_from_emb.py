#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crea carpetas 'train/' y 'val/' a partir de 'training/' sin re-extracción.
- Usa meta.csv para mapear unique_id -> rel_path (p.ej. training/positive/000123.mp4)
- Linkea (o copia) los embeddings ya extraídos:
    EMB_ROOT/training/{positive,negative}/000123.{npz|npy|pt|*}
  a:
    EMB_ROOT/train/{positive,negative}/...
    EMB_ROOT/val/{positive,negative}/...

Requisitos: Python 3.8+, sin deps externas.

Ejemplo:
  python utils/make_train_val_from_training_embeddings.py \
    --emb-root data_exp/vjepa2_l16s1_f20_hist5/dad \
    --meta data/metadata/DAD/dad_meta.csv \
    --train-ids data/metadata/DAD/splits/train_ids.txt \
    --val-ids   data/metadata/DAD/splits/val_ids.txt \
    --src-split training --out-train train --out-val val \
    --mode symlink --overwrite
"""
from __future__ import annotations
import argparse, csv, os, shutil
from pathlib import Path
from typing import Dict, List, Tuple

def read_ids(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

def load_meta(meta_csv: Path) -> Dict[str, Tuple[str, str]]:
    """
    Devuelve: uid -> (subdir, stem)
    Donde subdir ∈ {"positive","negative"} y stem como '000123'
    Se apoya en 'rel_path' del meta (p.ej. 'training/positive/000123.mp4').
    """
    out: Dict[str, Tuple[str, str]] = {}
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        R = csv.DictReader(f)
        for row in R:
            uid = row["unique_id"]
            rel = row["rel_path"]  # training/positive/000123.mp4
            parts = rel.split("/")
            if len(parts) < 3:
                continue
            subdir = parts[1]  # 'positive' o 'negative'
            stem = Path(parts[-1]).stem
            out[uid] = (subdir, stem)
    return out

def ensure_dirs(base: Path, subdirs: List[str]):
    for sd in subdirs:
        (base / sd).mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool):
    if dst.exists():
        if overwrite:
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        # Enlace simbólico (sirve para archivos o directorios)
        os.symlink(src, dst)
    elif mode == "hardlink":
        if src.is_dir():
            raise ValueError("hardlink no soporta directorios; usa symlink o copy")
        os.link(src, dst)
    else:
        # copy
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

def collect_sources(src_dir: Path, stem: str) -> List[Path]:
    """
    Intenta cubrir varios layouts de extracción:
      - archivos:  stem.{npz|npy|pt|*}
      - carpetas:  stem/  (con contenido dentro)
    """
    files = list(src_dir.glob(f"{stem}.*"))
    dirs  = [src_dir / stem] if (src_dir / stem).exists() and (src_dir / stem).is_dir() else []
    # Si no encontró nada con comodín, intenta algunos sufijos comunes:
    if not files and not dirs:
        for ext in (".npz", ".npy", ".pt"):
            p = src_dir / f"{stem}{ext}"
            if p.exists():
                files.append(p)
    return dirs + files  # prioriza directorios (si existen)

def process_split(uids: List[str], mapping: Dict[str, Tuple[str,str]],
                  emb_root: Path, src_split: str, out_split: str,
                  mode: str, overwrite: bool) -> Tuple[int,int]:
    """
    Crea out_split/{positive,negative} con enlaces/copias desde src_split/{...}
    """
    out_pos = emb_root / out_split / "positive"
    out_neg = emb_root / out_split / "negative"
    ensure_dirs(emb_root / out_split, ["positive", "negative"])

    n_ok, n_miss = 0, 0
    for uid in uids:
        if uid not in mapping:
            n_miss += 1
            continue
        subdir, stem = mapping[uid]
        src_dir = emb_root / src_split / subdir
        if not src_dir.exists():
            n_miss += 1
            continue
        candidates = collect_sources(src_dir, stem)
        if not candidates:
            n_miss += 1
            continue
        # Puede haber varios archivos (o un directorio). Replica todos.
        for src in candidates:
            relname = src.name  # mantiene nombre 'stem.ext' o carpeta 'stem'
            dst_base = out_pos if subdir == "positive" else out_neg
            dst = dst_base / relname
            try:
                # usar rutas absolutas para evitar líos de niveles
                link_or_copy(src.resolve(), dst, mode=mode, overwrite=overwrite)
                n_ok += 1
            except Exception as e:
                print(f"[WARN] No se pudo linkear/copiar {src} -> {dst}: {e}")
                n_miss += 1
    return n_ok, n_miss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-root", required=True, help="Raíz de embeddings (contiene 'training' y 'testing')")
    ap.add_argument("--meta", required=True, help="meta.csv con columnas unique_id, rel_path, ...")
    ap.add_argument("--train-ids", required=True, help="TXT con unique_id para train")
    ap.add_argument("--val-ids",   required=True, help="TXT con unique_id para val")
    ap.add_argument("--src-split", default="training", help="Nombre del split fuente (por defecto 'training')")
    ap.add_argument("--out-train", default="train",    help="Nombre de la carpeta de salida para train")
    ap.add_argument("--out-val",   default="val",      help="Nombre de la carpeta de salida para val")
    ap.add_argument("--mode", choices=["symlink","hardlink","copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true", help="Sobrescribe enlaces/archivos existentes")
    args = ap.parse_args()

    emb_root = Path(args.emb_root)
    meta_map = load_meta(Path(args.meta))
    train_uids = read_ids(Path(args.train_ids))
    val_uids   = read_ids(Path(args.val_ids))

    print(f"[INFO] emb_root = {emb_root}")
    print(f"[INFO] src_split= {args.src_split} | out_train= {args.out_train} | out_val= {args.out_val}")
    print(f"[INFO] mode     = {args.mode} | overwrite={args.overwrite}")
    print(f"[INFO] #train_uids={len(train_uids)} | #val_uids={len(val_uids)}")

    ok_tr, miss_tr = process_split(train_uids, meta_map, emb_root, args.src_split, args.out_train, args.mode, args.overwrite)
    ok_va, miss_va = process_split(val_uids,   meta_map, emb_root, args.src_split, args.out_val,   args.mode, args.overwrite)

    print(f"[DONE] train: ok={ok_tr} miss={miss_tr} -> {emb_root/args.out_train}")
    print(f"[DONE] val:   ok={ok_va} miss={miss_va} -> {emb_root/args.out_val}")

if __name__ == "__main__":
    main()
