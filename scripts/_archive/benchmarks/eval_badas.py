#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_badas.py — Script FINAL para Benchmark SOTA (AP/AUC + tiempos).

Calcula AP y AUC (protocolo ANYTIME) sobre Nexar:
  - Usa BADAS-Open como caja negra (model.predict(video_path) -> scores[t]).
  - No calcula mTTA (faltan timestamps de test).
  - Lee solution.csv oficial con columnas: id, target, Usage.
  - Soporta splits: public / private / both.

NOVEDADES vs versión simple:
  - Auto-detección robusta de la raíz del TFM (tanto layout V-JEPA 2 como ander/data/...).
  - Carga BADAS desde el repo local (badas) o, si falla, desde Hugging Face (badas_loader.py).
  - Cuenta número de parámetros y tamaño aproximado del modelo (MiB) leyendo el checkpoint.
  - Mide tiempo de inferencia por vídeo (promedio y primer vídeo) por split.
  - Checkpoints periódicos de predicciones en predictions/.
"""

import os, sys, time, csv, argparse
from pathlib import Path
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ======================================================================
#                AUTO-DETECCIÓN DE RUTAS (TFM / DATA)
# ======================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()

def find_tfm_root(start_path: Path, max_levels: int = 5) -> Optional[Path]:
    """Intenta localizar la raíz del TFM.

    Casos soportados:
      1) Layout antiguo:   .../V-JEPA 2/{data, scripts, ...}
      2) Layout nuevo:     .../ander/{data/metadata/Nexar, data/raw/Nexar, scripts}
    """
    current = start_path
    for _ in range(max_levels):
        # Caso 1: carpeta 'V-JEPA 2' colgando del padre
        vjepa = current.parent / "V-JEPA 2"
        if vjepa.exists() and vjepa.is_dir():
            return vjepa

        # Caso 2: este directorio ya es la raíz (tiene data/metadata/Nexar/solution.csv)
        if (current / "data/metadata/Nexar/solution.csv").exists():
            return current

        current = current.parent

    return None

PATH_TO_TFM = find_tfm_root(SCRIPT_DIR)
if PATH_TO_TFM is None:
    # Último fallback: edítalo a mano si hace falta
    PATH_TO_TFM = Path.home() / "ander"
    if not (PATH_TO_TFM / "data/metadata/Nexar/solution.csv").exists():
        print("[ERR] No se encuentra la raíz del TFM (ni V-JEPA 2 ni ander/data/...).")
        print("      Edita PATH_TO_TFM manualmente en eval_badas.py.")
        sys.exit(1)

print(f"[OK] ROOT TFM: {PATH_TO_TFM}")

PATH_TO_METADATA = PATH_TO_TFM / "data/metadata/Nexar"
PATH_TO_SOLUTION = PATH_TO_METADATA / "solution.csv"
PATH_TO_VIDEOS_BASE = PATH_TO_TFM / "data/raw/Nexar"
PATH_TO_SCRIPTS = PATH_TO_TFM / "scripts"
METRICS_FILE = PATH_TO_SCRIPTS / "metrics.py"

PRED_DIR = Path("predictions")
PRED_DIR.mkdir(exist_ok=True)

CHECKPOINT_EVERY = 20
EXT_CANDIDATES = [".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".avi"]
BADAS_FPS = 8.0  # El modelo BADAS opera a 8 FPS (internamente); aquí solo medimos wall-clock.

# ======================================================================
#                              UTILIDADES
# ======================================================================
def resolve_video_path(base_dir: Path, vid: str) -> Optional[Path]:
    """Busca un vídeo dentro de base_dir probando varias extensiones y subcarpetas."""
    # 1) Nombre directo (con o sin extensión)
    if any(vid.endswith(e) for e in EXT_CANDIDATES):
        direct = base_dir / vid
    else:
        direct = base_dir / f"{vid}.mp4"

    if direct.exists():
        return direct

    # 2) Otras extensiones en la raíz
    for ext in EXT_CANDIDATES:
        p = base_dir / f"{vid}{ext}"
        if p.exists():
            return p

    # 3) Búsqueda recursiva por subcarpetas
    for ext in EXT_CANDIDATES:
        hits = glob(str(base_dir / f"**/{vid}{ext}"), recursive=True)
        if hits:
            return Path(hits[0])

    return None

def _float_or_none(x: Any) -> Optional[float]:
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None

def read_solution_csv(
    csv_path: Path,
    usage_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Lee solution.csv (id, target, Usage, group) y aplica un filtro opcional por Usage."""
    out: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fns = r.fieldnames or []

        req_cols = {"id", "target"}
        if not req_cols.issubset(set(fns)):
            raise ValueError(f"solution.csv debe tener 'id' y 'target'. Encontradas: {fns}")

        has_usage = "Usage" in fns

        for row in r:
            vid = str(row["id"]).strip()
            target = int(float(row["target"]))
            usage = str(row.get("Usage", "")).strip() if has_usage else None
            group = str(row.get("group", "")).strip()

            if usage_filter is not None:
                if usage is None:
                    continue
                if usage.lower() != usage_filter.lower():
                    continue

            out.append({
                "id": vid,
                "target": target,
                "usage": usage,
                "group": group
            })

    return out

# ======================================================================
#                         IMPORT metrics.py (TFM)
# ======================================================================
if not METRICS_FILE.exists():
    print(f"[WARN] No se encuentra metrics.py en {METRICS_FILE}")
    metrics = None
else:
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("tfm_metrics", str(METRICS_FILE))
        metrics = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(metrics)
        print(f"[OK] metrics.py importado correctamente")
    except Exception as e:
        print(f"[WARN] Error importando metrics.py: {e}")
        metrics = None

# ======================================================================
#                    FALLBACKS NumPy para AP / AUC
# ======================================================================
def average_precision_np(y_true: np.ndarray, y_scores: np.ndarray) -> float:
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

# ======================================================================
#                             IMPORT BADAS
# ======================================================================
# 1) Intentar importar 'badas' desde el repo local BADAS-OPEN
BADAS_REPO_ROOT = SCRIPT_DIR.parent  # .../BADAS-OPEN
if (BADAS_REPO_ROOT / "badas").exists() and str(BADAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(BADAS_REPO_ROOT))

BADASModel = None
_load_badas_from_hf = None

try:
    from badas import BADASModel as _BADASModel  # type: ignore
    BADASModel = _BADASModel
    print("[OK] 'badas' importado desde el repo local BADAS-OPEN")
except Exception as e:
    print(f"[WARN] No se pudo importar 'badas' localmente: {e}")
    BADASModel = None

    # Fallback: loader oficial desde Hugging Face (badas_loader.py)
    try:
        from huggingface_hub import hf_hub_download

        def _load_badas_from_hf() -> Any:
            print("[INFO] Usando loader HF 'badas_loader.py' (nexar-ai/BADAS-Open)...")
            loader_path = hf_hub_download(
                repo_id="nexar-ai/BADAS-Open",
                filename="badas_loader.py",
            )
            loader_dir = os.path.dirname(loader_path)
            if loader_dir not in sys.path:
                sys.path.insert(0, loader_dir)
            from badas_loader import load_badas_model  # type: ignore
            return load_badas_model()

        print("[OK] Loader HF preparado (badas_loader.py)")
    except Exception as e2:
        print(f"[ERR] Tampoco se pudo preparar el loader HF de BADAS-Open: {e2}")
        _load_badas_from_hf = None

# ======================================================================
#             CONTEO DE PARÁMETROS Y TAMAÑO DEL MODELO (MiB)
# ======================================================================
def count_model_params_and_size() -> Tuple[int, float]:
    """Cuenta parámetros leyendo el checkpoint oficial de BADAS-Open.

    No depende de que el objeto `model` sea nn.Module; trabaja sobre el .pth.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print(f"[WARN] No se puede usar huggingface_hub para contar parámetros: {e}")
        return 0, float("nan")

    try:
        ckpt_path = hf_hub_download(
            repo_id="nexar-ai/BADAS-Open",
            filename="weights/badas_open.pth",
        )
    except Exception as e:
        print(f"[WARN] No se pudo descargar/encontrar el checkpoint BADAS-Open: {e}")
        return 0, float("nan")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] No se pudo cargar el checkpoint BADAS-Open: {e}")
        return 0, float("nan")

    if not isinstance(ckpt, dict):
        print("[WARN] Formato de checkpoint inesperado; no es un dict.")
        return 0, float("nan")

    # Buscar state_dict dentro de distintas claves habituales
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        # Asumir que el dict ya es un state_dict plano
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    total_params = 0
    total_bytes = 0
    for v in state_dict.values():
        if not isinstance(v, torch.Tensor):
            continue
        numel = v.numel()
        total_params += numel
        total_bytes += numel * v.element_size()

    size_mb = total_bytes / (1024 ** 2) if total_bytes > 0 else float("nan")
    return int(total_params), float(size_mb)

# ======================================================================
#                              EVALUACIÓN
# ======================================================================
def evaluate_split(
    split_name: str,
    rows: List[Dict[str, Any]],
    model: Any,
    video_base_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Evalúa un split (public / private) con protocolo ANYTIME.

    - Para cada vídeo, llama model.predict(path) -> secuencia de scores.
    - Usa max(score_t) como score ANYTIME por vídeo.
    - Calcula AP/AUC al final.
    - Mide tiempos de inferencia promedio por vídeo.
    """
    print("\n" + "=" * 70)
    print(f"📊 Evaluando split: {split_name.upper()}")
    print("=" * 70)
    print(f"Videos en split: {len(rows)}")

    missing_ids: List[str] = []
    jobs: List[Dict[str, Any]] = []

    for r in rows:
        vid = r["id"]
        path = resolve_video_path(video_base_dir, vid)

        if path is None:
            missing_ids.append(vid)
        else:
            jobs.append({"id": vid, "target": r["target"], "path": path})

    missing = len(missing_ids)
    print(f"Videos a procesar: {len(jobs)}")

    if missing > 0:
        print(f"[WARN] Videos ausentes: {missing}")
        ts = time.strftime("%Y%m%d-%H%M%S")
        miss_file = PRED_DIR / f"missing_{split_name}_{ts}.txt"
        with open(miss_file, "w") as f:
            for vid in missing_ids:
                f.write(f"{vid}\n")
        print(f"       Lista: {miss_file}")

    ids: List[str] = []
    scores_anytime: List[float] = []
    targets: List[int] = []

    ts_run = time.strftime("%Y%m%d-%H%M%S")
    out_csv = PRED_DIR / f"badas_nexar_{split_name}_{ts_run}.csv"

    def save_checkpoint():
        if not ids:
            return
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "score_anytime", "target"])
            for vid, s, t in zip(ids, scores_anytime, targets):
                w.writerow([vid, f"{s:.6f}", t])
        print(f"[CKPT] {len(ids)} predicciones → {out_csv}")

    # Barra de progreso
    try:
        from tqdm import tqdm
        iterator = tqdm(jobs, desc=f"BADAS-{split_name}", unit="vid")
        use_tqdm = True
    except ImportError:
        iterator = jobs
        use_tqdm = False

    processed = 0
    total_pred_time = 0.0
    first_video_time: Optional[float] = None
    first_video_id: Optional[str] = None
    
    try:
        for j in iterator:
            vid, target, path = j["id"], j["target"], j["path"]

            try:
                # Sincronizamos antes de medir (por si hay kernels pendientes)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                pred_seq = model.predict(str(path))

                # Sincronizamos después para que el tiempo incluya toda la inferencia
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

            except Exception as e:
                msg = f"[WARN] Error en {vid}: {e}"
                if use_tqdm:
                    iterator.write(msg)
                else:
                    print(msg)
                continue

            if pred_seq is None or len(pred_seq) == 0:
                continue

            arr = np.asarray(pred_seq, dtype=float)
            arr_valid = arr[np.isfinite(arr)]

            if arr_valid.size == 0:
                continue

            score_max = float(arr_valid.max())  # Protocolo ANYTIME

            scores_anytime.append(score_max)
            targets.append(target)
            ids.append(vid)
            processed += 1

            total_pred_time += dt
            if first_video_time is None:
                first_video_time = dt
                first_video_id = vid

            if use_tqdm:
                iterator.set_postfix({"score": f"{score_max:.3f}", "dt_s": f"{dt:.2f}"})

            if processed % CHECKPOINT_EVERY == 0:
                save_checkpoint()

    except KeyboardInterrupt:
        print("\n[INT] Interrumpido por el usuario.")
        save_checkpoint()
        return None


    save_checkpoint()
    print(f"[OK] Procesados: {len(scores_anytime)}")

    if len(scores_anytime) == 0:
        print("[ERR] No se han procesado vídeos válidos.")
        return None

    # Métricas
    print(f"Calculando métricas para {len(scores_anytime)} vídeos...")
    results: Dict[str, Any] = {}
    method = "NumPy fallback"

    y_scores = np.array(scores_anytime, dtype=np.float32)
    y_true = np.array(targets, dtype=np.int32)

    try:
        if metrics is not None and hasattr(metrics, "ap_auc_video_lit"):
            ap, auc = metrics.ap_auc_video_lit(y_true, y_scores)
            method = "metrics.ap_auc_video_lit"
        else:
            ap = average_precision_np(y_true, y_scores)
            auc = roc_auc_np(y_true, y_scores)
    except Exception as e:
        print(f"[ERR] Fallo calculando AP/AUC: {e}")
        ap, auc = np.nan, np.nan

    results["AP"] = float(ap)
    results["AUC"] = float(auc)
    results["Method_AP_AUC"] = method
    results["videos_evaluados"] = int(len(scores_anytime))
    results["videos_ausentes"] = int(missing)

    # Tiempos
    avg_time = total_pred_time / processed if processed > 0 else float("nan")
    results["infer_total_s"] = float(total_pred_time)
    results["infer_avg_s_per_video"] = float(avg_time)
    results["infer_first_s"] = float(first_video_time) if first_video_time is not None else float("nan")
    results["infer_first_id"] = first_video_id

    print("-" * 60)
    print(f"⏱️ Tiempo TOTAL inferencia split '{split_name}': {total_pred_time:.3f} s")
    print(f"⏱️ Tiempo MEDIO por vídeo: {avg_time:.3f} s ({processed} vídeos)")
    if first_video_time is not None:
        print(f"⏱️ Primer vídeo ({first_video_id}) tardó: {first_video_time:.3f} s")
    print("-" * 60)

    return results

# ======================================================================
#                                 MAIN
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="public",
        choices=["public", "private", "both"],
        help="Split a evaluar (default: public)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 EVALUACIÓN BADAS-Open en Nexar (AP/AUC + tiempos)")
    print("=" * 70)

    if not PATH_TO_SOLUTION.exists():
        print(f"[ERR] No existe solution.csv en {PATH_TO_SOLUTION}")
        return

    print(f"CSV:   {PATH_TO_SOLUTION}")
    print(f"Split: {args.split.upper()}")

    # Cargar modelo BADAS-Open
    print("\n[INFO] Cargando BADAS-Open...")
    if BADASModel is not None:
        model = BADASModel(device="cuda")
    elif _load_badas_from_hf is not None:
        model = _load_badas_from_hf()
    else:
        print("[ERR] No hay forma de cargar BADAS-Open (ni 'badas' ni loader HF disponibles).")
        return
    print("[OK] Modelo BADAS cargado.\n")

    # Parámetros y tamaño aproximado
    total_params, size_mb = count_model_params_and_size()
    if total_params > 0:
        print(f"[INFO] BADAS-Open → parámetros: {total_params:,d}")
        print(f"[INFO] BADAS-Open → tamaño aprox.: {size_mb:.1f} MiB (solo pesos)")
    else:
        print("[WARN] No se pudieron estimar parámetros/tamaño del modelo.")

    # Leer solution.csv
    all_rows = read_solution_csv(PATH_TO_SOLUTION, usage_filter=None)
    print(f"\n[INFO] solution.csv: {len(all_rows)} vídeos totales")

    public_rows = [r for r in all_rows if r["usage"] == "Public"]
    private_rows = [r for r in all_rows if r["usage"] == "Private"]
    print(f"       Public:  {len(public_rows)}")
    print(f"       Private: {len(private_rows)}")

    final_results: Dict[str, Optional[Dict[str, Any]]] = {}

    if args.split in {"public", "both"}:
        video_dir = PATH_TO_VIDEOS_BASE / "test-public"
        if not video_dir.exists():
            print(f"[WARN] No existe {video_dir}, probando en la raíz de vídeos...")
            video_dir = PATH_TO_VIDEOS_BASE

        res_pub = evaluate_split("public", public_rows, model, video_dir)
        final_results["public"] = res_pub

    if args.split in {"private", "both"}:
        video_dir = PATH_TO_VIDEOS_BASE / "test-private"
        if not video_dir.exists():
            print(f"[WARN] No existe {video_dir}, probando en la raíz de vídeos...")
            video_dir = PATH_TO_VIDEOS_BASE

        res_priv = evaluate_split("private", private_rows, model, video_dir)
        final_results["private"] = res_priv

    # Resumen final
    print("\n" + "=" * 70)
    print("📈 RESULTADOS FINALES (AP/AUC + tiempos)")
    print("=" * 70)

    for split_name, res in final_results.items():
        if res is None:
            print(f"\n[{split_name.upper()}] Sin datos válidos")
            continue

        print(f"\n[{split_name.upper()}]")
        print(f"  Videos evaluados: {res.get('videos_evaluados', 0)}")
        print(f"  Videos ausentes:  {res.get('videos_ausentes', 0)}")
        print("-" * 20)
        print(f"  AP (Anytime):   {res.get('AP', np.nan):.6f}")
        print(f"  AUC (Anytime):  {res.get('AUC', np.nan):.6f}")
        print("-" * 20)
        print(f"  Método AP/AUC:  {res.get('Method_AP_AUC', 'N/A')}")
        print("-" * 20)
        print(f"  Tiempo total inferencia: {res.get('infer_total_s', float('nan')):.3f} s")
        print(f"  Tiempo medio / vídeo:   {res.get('infer_avg_s_per_video', float('nan')):.3f} s")
        first_id = res.get("infer_first_id", None)
        first_s = res.get("infer_first_s", float("nan"))
        if first_id is not None:
            print(f"  Primer vídeo ({first_id}) tardó: {first_s:.3f} s")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
