#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import shutil

# === CONFIG ===
# Directorio donde están los tokens originales
SRC_ROOT = Path("data_exp/vjepa2_tokens/Nexar")

# Nueva raíz con estructura positive/negative compatible con tu pipeline DAD/CCD
DST_ROOT = Path("data_exp/vjepa2_tokens/Nexar_split")

# Directorio donde están los CSV de Nexar que usaste para extracción
CSV_ROOT = Path("data/metadata/Nexar/extraction_csvs")

ID_COL = "id"   # cambia a "video_id" si tu CSV usa otro nombre

def id_to_fname(v) -> str:
    """
    Convierte el valor de la columna id del CSV al nombre del .pt:
    3       -> "00003.pt"
    188.0   -> "00188.pt"
    "42"    -> "00042.pt"
    """
    try:
        vid_int = int(v)
    except Exception:
        # fallback brutal por si viene como string raro
        vid_int = int(str(v).split(".")[0])
    return f"{vid_int:05d}.pt"


def split_split(split: str):
    """
    split ∈ {"train", "val"}
    Crea:
      Nexar_split/{split}/training/positive
      Nexar_split/{split}/training/negative
    a partir de CSV_ROOT/{split}.csv
    """
    csv_path = CSV_ROOT / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe CSV: {csv_path}")

    print(f"[INFO] Leyendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if ID_COL not in df.columns:
        raise KeyError(f"La columna '{ID_COL}' no está en {csv_path}. Columnas: {df.columns.tolist()}")

    if "target" not in df.columns:
        raise KeyError(f"Falta columna 'target' en {csv_path} (necesaria para pos/neg).")

    src_dir = SRC_ROOT / split
    if not src_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de tokens: {src_dir}")

    n_tot = 0
    n_ok = 0
    n_missing = 0

    for _, row in df.iterrows():
        n_tot += 1
        vid = row[ID_COL]
        target = int(row["target"])

        cls = "positive" if target == 1 else "negative"
        dst_dir = DST_ROOT / split / "training" / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        fname = id_to_fname(vid)  # e.g. "00188.pt"
        src = src_dir / fname

        if not src.exists():
            print(f"[WARN] No .pt para id={vid} -> {fname} en {src_dir}")
            n_missing += 1
            continue

        dst = dst_dir / fname

        if dst.exists():
            # ya creado (por si re-ejecutas)
            n_ok += 1
            continue

        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

        n_ok += 1

    print(f"[DONE] split={split}: total={n_tot}, enlazados/copias={n_ok}, missing={n_missing}")


if __name__ == "__main__":
    for split in ["train", "val"]:
        split_split(split)
    print("✅ Nexar separado en positive/negative en data_exp/vjepa2_tokens/Nexar_split/")
