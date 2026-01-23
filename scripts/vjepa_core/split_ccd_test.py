#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toma una carpeta 'src' con muchos archivos .pt y los divide en subcarpetas
'dst/positive' y 'dst/negative' usando un .csv de metadatos.

VERSIÓN 3.0: Corrige bug donde 'rel_path' en el CSV contenía una
extensión (ej. 'video.mp4') que corrompía el nombre del .pt final.
"""
import argparse
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser("Divide una carpeta de .pt en subcarpetas pos/neg")
    ap.add_argument("--meta", required=True, help="Ruta al .csv de metadatos (ej. ccd_meta.csv)")
    ap.add_argument("--src", required=True, help="Carpeta 'source' con todos los .pt (ej. .../ccd/test)")
    ap.add_argument("--dst", required=True, help="Carpeta 'destination' (ej. .../ccd/test/testing)")
    ap.add_argument("--ext", default=".pt", help="Extensión de los archivos a mover (ej. .pt)")
    
    # Columnas del CSV
    ap.add_argument("--split-col", required=True, help="Columna del CSV que define el split (ej. 'split')")
    ap.add_argument("--split-value", required=True, help="Valor en split_col a procesar (ej. 'test')")
    ap.add_argument("--relpath-col", required=True, help="Columna del CSV con el nombre/ruta relativa (ej. 'rel_path')")
    ap.add_argument("--target-col", default="target", help="Columna del CSV con la etiqueta 0 o 1 (ej. 'target')")
    
    args = ap.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    meta_path = Path(args.meta).resolve()

    if not meta_path.exists():
        print(f"❌ Error: No se encuentra el CSV de metadatos en {meta_path}")
        return
    
    if not src_dir.exists():
        print(f"❌ Error: No se encuentra la carpeta 'src' en {src_dir}")
        return

    # Crear carpetas de destino
    dst_pos = dst_dir / "positive"
    dst_neg = dst_dir / "negative"
    dst_pos.mkdir(parents=True, exist_ok=True)
    dst_neg.mkdir(parents=True, exist_ok=True)

    print("==== SUMMARY ====")
    print(f"SRC      : {src_dir}")
    print(f"DST      : {dst_dir}")
    print(f"EXT      : {args.ext}")
    print(f"META     : {meta_path.name}")

    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        print(f"❌ Error al leer {meta_path}: {e}")
        return

    df_split = df[df[args.split_col] == args.split_value].copy()
    
    if df_split.empty:
        print(f"❌ Error: No se encontraron filas con {args.split_col} == '{args.split_value}' en el CSV.")
        return

    count_pos = 0
    count_neg = 0
    count_miss = 0

    print(f"\nMoviendo {len(df_split)} archivos de {args.split_value}...")
    
    for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
        try:
            # ¡¡ESTE ES EL DATO QUE USAMOS PARA EL NOMBRE DE ARCHIVO!!
            rel_path_from_csv = str(row[args.relpath_col]) 
            target = int(row[args.target_col])
        except Exception as e:
            print(f"Error en fila de CSV (saltando): {e} | Fila: {row}")
            continue
            
        # === LA CORRECCIÓN ES AQUÍ ===
        # 1. Coge el nombre base (ej. 'videos/C-000003.mp4' -> 'C-000003.mp4')
        basename = Path(rel_path_from_csv).name
        # 2. Coge solo el 'stem' (el nombre sin extensión, ej. 'C-000003')
        stem = Path(basename).stem
        # 3. Añade la extensión .pt (ej. 'C-000003.pt')
        filename = stem + args.ext
        
        src_file = src_dir / filename
        # ==============================

        if not src_file.exists():
            # print(f"WARN: No se encontró {src_file}")
            count_miss += 1
            continue

        if target == 1:
            dst_file = dst_pos / filename
            count_pos += 1
        else:
            dst_file = dst_neg / filename
            count_neg += 1
        
        try:
            shutil.move(str(src_file), str(dst_file))
        except Exception as e:
            print(f"Error moviendo {src_file} a {dst_file}: {e}")

    print("\n=================")
    print(f"COPIADOS : positive={count_pos}, negative={count_neg}")
    if count_miss > 0:
        print(f"FALTANTES: {count_miss} (archivos en .csv pero no encontrados en {src_dir})")
    print("=================")

if __name__ == "__main__":
    main()