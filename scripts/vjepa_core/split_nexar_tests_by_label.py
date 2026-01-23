#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_nexar_tests_by_label.py

Reorganiza los .pt de Nexar TEST (public/private) en carpetas positive/negative,
usando el CSV de extracción (test_public.csv o test_private.csv).

Asume formato CSV tipo:
    video_id, t_event, t_cut, label

Ejemplo:
    01248,,,0
    00529,19.805,18.629,1

Uso típico:

  # TEST PUBLIC
  ~/.local/bin/poetry run python scripts/split_nexar_tests_by_label.py \
    --src-root data_exp/vjepa2_tokens/Nexar/test_public \
    --dst-root data_exp/vjepa2_tokens/Nexar_split/test_public/testing \
    --csv data/metadata/Nexar/extraction_csvs/test_public.csv

  # TEST PRIVATE
  ~/.local/bin/poetry run python scripts/split_nexar_tests_by_label.py \
    --src-root data_exp/vjepa2_tokens/Nexar/test_private \
    --dst-root data_exp/vjepa2_tokens/Nexar_split/test_private/testing \
    --csv data/metadata/Nexar/extraction_csvs/test_private.csv
"""

import argparse
import csv
from pathlib import Path
import shutil


def load_labels_from_csv(csv_path: Path) -> dict[str, int]:
    """
    Carga {video_id: label(0/1)} desde el CSV.

    Asume:
      col 0 = video_id
      col 3 = label (0/1)  -> formato id, t_event, t_cut, y
    """
    labels: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            vid = row[0].strip()
            if not vid:
                continue

            y = None
            # Intento principal: 4ª columna
            if len(row) >= 4 and row[3].strip() != "":
                try:
                    y = int(row[3])
                except ValueError:
                    y = None

            # Fallback: 3ª columna si está en {0,1}
            if y is None and len(row) >= 3 and row[2].strip() in ("0", "1"):
                y = int(row[2].strip())

            if y is None or y not in (0, 1):
                continue

            labels[vid] = y

    if not labels:
        raise SystemExit(f"❌ No se pudieron extraer etiquetas de {csv_path}")
    print(f"[INFO] Cargadas {len(labels)} etiquetas desde {csv_path}")
    return labels


def infer_video_id(stem: str, labels: dict[str, int]) -> str | None:
    """
    Mapea nombre de archivo (.pt) -> video_id del CSV.

    Estrategia:
      1) stem tal cual (p.ej. '00001')
      2) stem antes de '__'
      3) primer token antes de '_' (por si metes sufijos)
    """
    candidates = [stem]

    if "__" in stem:
        candidates.append(stem.split("__", 1)[0])
    if "_" in stem:
        candidates.append(stem.split("_", 1)[0])

    for c in candidates:
        if c in labels:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-root",
        type=str,
        required=True,
        help="Directorio con los .pt SIN dividir (p.ej. Nexar/test_public)",
    )
    ap.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Directorio base de salida (se crean subcarpetas positive/negative)",
    )
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV de extracción (test_public.csv o test_private.csv)",
    )
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    csv_path = Path(args.csv).resolve()

    if not src_root.exists():
        raise SystemExit(f"❌ src_root no existe: {src_root}")
    if not csv_path.exists():
        raise SystemExit(f"❌ CSV no existe: {csv_path}")

    labels = load_labels_from_csv(csv_path)

    # Crear dst_root y subcarpetas
    dst_root.mkdir(parents=True, exist_ok=True)
    pos_dir = dst_root / "positive"
    neg_dir = dst_root / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Recorremos solo los .pt directamente en src_root
    pts = sorted([p for p in src_root.glob("*.pt") if p.is_file()])
    if not pts:
        print(f"[WARN] No hay .pt directamente en {src_root} (¿ya están divididos?).")
        return

    n_pos = n_neg = n_unmatched = 0

    for p in pts:
        stem = p.stem
        vid = infer_video_id(stem, labels)
        if vid is None:
            print(f"[WARN] Sin etiqueta para {p.name} (stem='{stem}')")
            n_unmatched += 1
            continue

        y = labels[vid]
        dst = pos_dir / p.name if y == 1 else neg_dir / p.name

        # mover (rename dentro del mismo FS)
        try:
            p.rename(dst)
        except OSError:
            shutil.move(str(p), str(dst))

        if y == 1:
            n_pos += 1
        else:
            n_neg += 1

    print("\n" + "=" * 70)
    print(f"[DONE] src_root: {src_root}")
    print(f"  Movidos a positive: {n_pos}")
    print(f"  Movidos a negative: {n_neg}")
    print(f"  Sin match en CSV:  {n_unmatched}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
