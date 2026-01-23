#!/usr/bin/env python3
"""
Crea los archivos train.csv y val.csv para el dataset Nexar.

1. Escanea las carpetas positive/ y negative/ en `videos_root`.
2. Crea una lista de todos los (video_id, label).
3. Divide esta lista en 80% train y 20% val (estratificado).
4. Guarda los archivos en `metadata_root` (definido en el yaml).
"""
import csv
import yaml
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Importa el loader del adapter
from nexar import load_config

def main():
    # Resolver relativo al repo para que funcione independientemente del cwd
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "configs" / "nexar.yaml"
    try:
        config = load_config(str(config_path))
    except FileNotFoundError:
        print(f"Error: No se encontró '{config_path}'.")
        print("Asegúrate de que 'nexar.yaml' esté en la carpeta 'configs/'.")
        return

    videos_root = Path(config['paths']['videos_root'])
    metadata_root = Path(config['paths']['metadata_root'])
    metadata_root.mkdir(parents=True, exist_ok=True)

    print(f"Buscando videos en: {videos_root}")
    
    pos_videos = list((videos_root / "positive").glob("*.mp4"))
    neg_videos = list((videos_root / "negative").glob("*.mp4"))

    print(f"Encontrados {len(pos_videos)} positivos y {len(neg_videos)} negativos.")
    
    if len(pos_videos) == 0:
        print("Error: No se encontraron videos. Verifica tu 'videos_root' en nexar.yaml.")
        return

    # Crear lista de (video_id, label)
    all_samples = []
    for p in pos_videos:
        all_samples.append({"video_id": p.name, "label": 1})
    for p in neg_videos:
        all_samples.append({"video_id": p.name, "label": 0})

    labels = [s['label'] for s in all_samples]

    # Dividir 80/20 estratificado
    train_samples, val_samples = train_test_split(
        all_samples, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )

    print(f"Dividiendo en {len(train_samples)} train y {len(val_samples)} val.")

    # Escribir CSVs
    fieldnames = ["video_id", "label"]
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        out_path = metadata_root / f"{split_name}.csv"
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)
        print(f"✅ Archivo '{split_name}.csv' guardado en {metadata_root}")

if __name__ == "__main__":
    main()
