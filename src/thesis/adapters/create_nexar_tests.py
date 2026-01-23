#!/usr/bin/env python3
"""
Lee 'solution.csv' del dataset Nexar y lo divide en
'test_public.csv' y 'test_private.csv', guardándolos
en la carpeta de metadatos.
"""
import csv
from pathlib import Path
import yaml

def main() -> int:
    # Cargar config para saber dónde están las carpetas (independiente del cwd)
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "configs" / "nexar.yaml"

    if not config_path.exists():
        print(f"Error: No se encuentra {config_path}")
        return 2

    with config_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    paths = config_data.get("paths", {}) or {}
    videos_root_raw = paths.get("videos_root", "")
    metadata_root_raw = paths.get("metadata_root", "")

    if not str(videos_root_raw).strip():
        print("Error: Falta paths.videos_root en configs/nexar.yaml")
        return 2
    if not str(metadata_root_raw).strip():
        print("Error: Falta paths.metadata_root en configs/nexar.yaml")
        return 2

    videos_root = Path(str(videos_root_raw)).expanduser()
    metadata_root = Path(str(metadata_root_raw)).expanduser()

    # Ruta donde están los videos de test (base de nexar_collision_prediction)
    # Asumimos que solution.csv está en la carpeta raíz del dataset
    raw_data_root = videos_root.parent

    solution_csv_path = raw_data_root / "solution.csv"
    if not solution_csv_path.exists():
        print(f"Error: No se encuentra {solution_csv_path}")
        print("Asegúrate de tener 'solution.csv' en la carpeta 'nexar_collision_prediction/'.")
        return 2

    print(f"Cargando {solution_csv_path}...")

    # Listas para guardar las filas
    public_samples: list[dict] = []
    private_samples: list[dict] = []

    # Leer el solution.csv
    # Formato esperado: video_id,label,split
    with solution_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = int(float(row.get("label", 0)))
            except Exception:
                continue
            sample = {"video_id": row.get("video_id"), "label": label}

            if row.get("split") == "test_public":
                public_samples.append(sample)
            elif row.get("split") == "test_private":
                private_samples.append(sample)

    print(f"Encontradas {len(public_samples)} muestras 'test_public'")
    print(f"Encontradas {len(private_samples)} muestras 'test_private'")

    metadata_root.mkdir(parents=True, exist_ok=True)

    # Escribir los nuevos CSVs
    fieldnames = ["video_id", "label"]

    # Guardar test_public.csv
    public_csv_path = metadata_root / "test_public.csv"
    with public_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_samples)
    print(f"✅ 'test_public.csv' guardado en {metadata_root}")

    # Guardar test_private.csv
    private_csv_path = metadata_root / "test_private.csv"
    with private_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(private_samples)
    print(f"✅ 'test_private.csv' guardado en {metadata_root}")

    print("\n¡División de test sets completada!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
