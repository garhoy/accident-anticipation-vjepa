#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanza varios entrenos de forma secuencial y compila un resumen.

CAMBIOS CRÍTICOS vs versión anterior:
1. Eliminados modelos redundantes (TCNFromTokensMean, FiLMTCNFromMean)
2. Hiperparámetros ajustados según análisis de sample-efficiency
3. Transformer con capacidad reducida para evitar overfitting
4. BADAS/tokens con regularización sensata
5. Documentación de cada decisión de diseño

Autor: [Tu nombre]
Fecha: 2025-11
"""

import sys, subprocess, json, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

from pathlib import Path
from typing import Dict, Any
import torch

EXPERIMENTS_ROOT = Path("experiments_dad")

# # ========= Rutas a embeddings / tokens =========
TRAIN_DIRS = [
    "data_exp/vjepa2_tokens/dad/train/training/positive",
    "data_exp/vjepa2_tokens/dad/train/training/negative",
]
VAL_DIRS = [
    "data_exp/vjepa2_tokens/dad/val/training/positive",
    "data_exp/vjepa2_tokens/dad/val/training/negative",
]

# IDs oficiales (opcional, para reproducibilidad exacta con BADAS paper)
TRAIN_IDS_FILE = "data/metadata/DAD/splits/train_ids.txt"
VAL_IDS_FILE   = "data/metadata/DAD/splits/val_ids.txt"
USE_OFFICIAL_IDS = False
DATASET_NAME = "dad"

# # Usa una raíz distinta para no pisar DAD
# EXPERIMENTS_ROOT = Path("experiments_ccd")

# TRAIN_DIRS = [
#     "data_exp/vjepa2_tokens/ccd/train/training/positive",
#     "data_exp/vjepa2_tokens/ccd/train/training/negative",
# ]
# VAL_DIRS = [
#     "data_exp/vjepa2_tokens/ccd/val/training/positive",
#     "data_exp/vjepa2_tokens/ccd/val/training/negative",
# ]
# TRAIN_IDS_FILE = "data/metadata/CCD/splits_ccd/train_ids.txt"
# VAL_IDS_FILE   = "data/metadata/CCD/splits_ccd/val_ids.txt"
# USE_OFFICIAL_IDS = False
# DATASET_NAME = "ccd"

# ========= Modelos CORE (justificados científicamente) =========
MODELS = [
    # EMBEDDINGS (baselines clásicos)
    # "gru",          # Memoria recurrente, baseline fuerte
    # "tcn",          # Convolución causal, sample-efficient
    # "transformer",  # Atención global (esperado: underperform con pocos datos)
    
    # TOKENS (explotan estructura espacial)
    "badas",            # Referencia del paper original
    "temporal_attn",    # Spatial pooling + Transformer temporal
    "deformable_event"
]

# ! CAMBIE EL EARLY STOPPING PARA DEFORMABLE 
# Conjunto de modelos que CONSUMEN TOKENS (necesitan --use-tokens)
TOKEN_MODELS = {
    "badas",
    "temporal_attn",
    "deformable_event"
}

# ========= Hiper generales =========
WINDOW_S = 5                # ventana de observación (segundos)
HORIZONS = [1, 2, 3, 4, 5]  # horizontes de predicción (segundos)
EPOCHS   = 50               # máximo (early stopping activo)
BATCH_DEFAULT_EMBED = 64    # embeddings caben más
BATCH_DEFAULT_TOKEN = 1     # tokens consumen mucha RAM
REPEATS = 1                # robustez estadística (mean ± std)
START_RUN_INDEX = 0
BALANCED_SAMPLER = True     # 50/50 pos/neg para clases desbalanceadas

# ========= Protocolo de evaluación =========
AP_SCOPE       = "anytime"  # comparable con CRASH/LATTE (también guarda "pre")
DECISION_POINT = "end"      # predicción al final del clip
EVENT_ALIGN    = "clip_end" # alineación temporal del evento

# ========= Path a train_once.py =========
SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPT_PATH = SCRIPT_DIR / "train_once.py"
if not SCRIPT_PATH.exists():
    SCRIPT_PATH = Path("scripts/train_once.py").resolve()

# ========================================================================
# HIPERPARÁMETROS POR MODELO (JUSTIFICADOS)
# ========================================================================
MODEL_KW: Dict[str, Dict[str, Any]] = {
    # =====================================================================
    # EMBEDDINGS (z_clip frame-wise)
    # =====================================================================
    
    "gru": {
        # GRU clásico: memoria recurrente eficiente
        # DECISIÓN: Config estándar, funciona bien out-of-the-box
        "hidden": 512,          # capacidad suficiente para D=1024
        "num_layers": 2,        # 2 capas = buen compromiso
        "dropout": 0.30,        # regularización moderada
        "bidirectional": False, # causal (online setting)
    },
    
    "tcn": {
        # TCN: receptive field causal grande, muy sample-efficient
        # DECISIÓN: Tu mejor baseline actual, mantener configuración probada
        "channels": 256,        # suficiente capacidad
        "levels": 5,            # RF ≈ 31 frames @ 20fps = 1.5s (adecuado)
        "k": 3,                 # kernel pequeño, RF crece con dilations
        "dropout": 0.30,        # ✅ REDUCIDO de 0.40 (estaba sobre-regularizando)
    },
    
    "transformer": {
        "d_model": 384,
        "n_heads": 6,
        "num_layers": 4,
        "ff_dim": 1536,
        "dropout": 0.10,
        "pe_dropout": 0.05,
        "attn_dropout": 0.05,
        "causal": True,
    },
    
    # =====================================================================
    # TOKENS (spatial tokens de V-JEPA)
    # =====================================================================
    
    "badas": {
        # BADAS: Configuración EXACTA del paper original
        # Paper: "BADAS: A Large-Scale Benchmark for Anticipating Accidents"
        # NO TOCAR - mantener fidelidad al paper para comparación justa
        "M": 12,                # queries aprendibles (paper original)
        "d": 64,                # dim proyección (paper original)
        "n_heads": 16,          # atención espacial (paper original)
        "num_attn_layers": 4,   # capas de atención (paper original)
        "dropout": 0.20,        # regularización MLP
        "attn_dropout": 0.10,   # regularización atención
        "upsample_to_frames": True,  # Dt -> T frames (necesario para protocolo)
        # NOTA: Config más grande que otros modelos, pero respeta paper
    },
    
    "temporal_attn": {
        # Spatial pooling (attentive) + Transformer temporal causal
        # DECISIÓN: Reducir capacidad y regularización menos agresiva
        "M": 6,                 # queries espaciales
        "proj_dim": 48,         # ✅ REDUCIDO de 64 (dim temporal)
        "n_heads_spatial": 8,   # ✅ REDUCIDO de 8 (pooler)
        "n_heads_temporal": 2,  # atención temporal
        "tfm_layers": 2,        # capas Transformer temporales
        "dropout": 0.20,        # ✅ REDUCIDO de 0.50 (era DEMASIADO agresivo)
        "attn_dropout": 0.10,   # atención
        "token_dropout_p": 0.10,    # stochastic depth espacial
        "temporal_dropout_p": 0.05, # stochastic depth temporal
        "freeze_pooler": False, # entrenar end-to-end
        # EXPECTATIVA: Competir con BADAS con menos parámetros
    },

    "deformable_event": {
        # Deformable Event-Query + Δ-bias (Tier-A)
        "n_queries": 12,
        "num_points": 12,
        "num_levels": 2,
        "num_layers": 3,
        "d_model": 256,
        "dropout": 0.10,
        "attn_dropout": 0.05,
        "delta_lambda": 0.5,
        "upsample_to_frames": True,
        "use_checkpoint": True,
        "align_corners": False
    },
}

# ========================================================================
# UTILIDADES
# ========================================================================

def preferred_dir_name(model_name: str) -> str:
    """Nombre de directorio estandarizado para experimentos."""
    w = str(WINDOW_S).replace('.', 'p')
    return f"batch_{model_name}_W{w}_{DECISION_POINT}_{EVENT_ALIGN}_{AP_SCOPE}_{DATASET_NAME}"

def candidate_dirs_for(model_name: str, run_idx: int):
    """Busca directorios de experimentos (nombres legacy + nuevos)."""
    base_name = preferred_dir_name(model_name)
    yield EXPERIMENTS_ROOT / f"{base_name}_{run_idx}"
    yield EXPERIMENTS_ROOT / f"{model_name}_{run_idx}"  # fallback

def run_one(model_name: str, run_idx: int) -> int:
    """
    Lanza un entrenamiento individual.
    Returns: 0 si éxito, 1 si fallo
    """
    is_token_model = model_name in TOKEN_MODELS
    batch_size = BATCH_DEFAULT_TOKEN if is_token_model else BATCH_DEFAULT_EMBED

    out_dir_name = preferred_dir_name(model_name)
    out_dir = EXPERIMENTS_ROOT / f"{out_dir_name}_{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_kw_json = json.dumps(MODEL_KW.get(model_name, {}))

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--model", model_name,
        "--train-dirs", *TRAIN_DIRS,
        "--val-dirs", *VAL_DIRS,
        "--horizons", *[str(h) for h in HORIZONS],
        "--window-s", str(WINDOW_S),
        "--epochs", str(EPOCHS),
        "--batch-size", str(batch_size),
        "--out-root", str(out_dir),
        "--ap-scope", AP_SCOPE,
        f"--run-idx={run_idx}",
        "--model-kw", model_kw_json,

        # ===== EARLY STOPPING =====
        "--early-stop-patience", "4",
        "--early-stop-min-delta", "0.002",
        "--min-epochs", "6",

        # ===== SCHEDULER: Cosine + Warmup =====
        "--cosine",
        "--warmup-epochs", "3",
    ]

    if BALANCED_SAMPLER:
        cmd.append("--balanced-sampler")
        cmd.append("--pos-weight")  # sampler + pos_weight (recomendado)

    if is_token_model:
        # ===== TOKENS: estabilidad/VRAM =====
        cmd.append("--use-tokens")
        cmd.extend(["--accum-steps", "8"])
        cmd.append("--amp")  # mixed precision
        # SIN EMA en tokens
    else:
        # ===== EMA solo en embeddings =====
        cmd.extend(["--ema", "--ema-decay", "0.999"])

    if USE_OFFICIAL_IDS:
        cmd.extend(["--train-ids", TRAIN_IDS_FILE, "--val-ids", VAL_IDS_FILE])

    print("\n" + "-"*70)
    print(f"🚀 Launching {model_name} (run {run_idx})")
    print(f"   Output: {out_dir}")
    if model_kw_json != "{}":
        print(f"   Config: {model_kw_json}")
    print("-"*70)

    # ===== Entorno para evitar OOM por fragmentación =====
    env = os.environ.copy()
    # env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Opcional: trazas útiles si crashea dentro de kernels
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

    log_file = out_dir / "train.log"
    process = None
    try:
        with log_file.open("w", encoding="utf-8") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,  # <<< usar env con config de memoria
            )
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                    f.write(line)
            process.wait()
            rc = process.returncode
    except KeyboardInterrupt:
        print(f"\n⚠️  [ABORT] Interrupción manual de {model_name}")
        if process:
            process.terminate()
        return 1
    except Exception as e:
        print(f"❌ [ERROR] Fallo al lanzar {model_name}: {e}")
        return 1

    if rc != 0 and log_file.exists():
        print(f"\n⚠️  [WARN] Entrenamiento falló (rc={rc})")
        try:
            print("\n" + "="*70)
            print("📋 Últimas 80 líneas de train.log:")
            print("="*70)
            tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            print("\n".join(tail))
            print("="*70 + "\n")
        except Exception:
            pass
    return rc


def load_summary_for(exp_dir: Path):
    """Extrae métricas del checkpoint best_model.pt."""
    ckpt_path = exp_dir / "best_model.pt"
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        valm = ckpt.get("val_metrics", {}) or {}
        
        # Fallback a val_history.json para mTTA/TTA (si no están en val_metrics)
        hist_path = exp_dir / "val_history.json"
        best_epoch_hist = {}
        if hist_path.exists():
            import json as _json
            with hist_path.open("r") as f:
                history = _json.load(f)
            best_epoch_hist = next(
                (e for e in history if e.get("epoch") == ckpt.get("epoch", -1)), 
                {}
            )
        
        ap = valm.get("AP_video_lit", None)
        auc = valm.get("AUC_video_lit", None)
        mtta = best_epoch_hist.get("mTTA_free_lit", None) or valm.get("mTTA_free_lit", None)
        tta_r80 = best_epoch_hist.get("TTA_R80", None) or valm.get("TTA_R80", None)
        
        return {
            "epoch": ckpt.get("epoch", None),
            "AP_video_lit": ap,
            "AUC_video_lit": auc,
            "mTTA_free_lit": mtta,
            "TTA_R80": tta_r80,
            "exp_dir": str(exp_dir.name),
        }
    except Exception as e:
        print(f"⚠️  [WARN] Error al leer {ckpt_path}: {e}")
        return None

def print_table(data: list[dict[str, Any]]):
    """Imprime tabla ordenada por AP descendente."""
    if not data:
        print("\n❌ No se encontraron resultados.")
        return
    
    print("\n" + "="*90)
    print("📊 RESUMEN DE RESULTADOS (best_model.pt)")
    print("="*90)
    
    # Ordenar por AP descendente
    data.sort(key=lambda r: (r.get("AP_video_lit") or 0.0), reverse=True)
    
    # Header
    hdr = (
        f"{'MODELO':<20} {'RUN':<4} {'EPOCH':<6} "
        f"{'AP_VID':<10} {'AUC_VID':<10} {'mTTA':<10} {'TTA@R80':<10}"
    )
    print(hdr)
    print("-" * 90)
    
    def _f(row, k: str, prec: int, default: str = "N/A") -> str:
        v = row.get(k)
        try:
            return default if v is None else f"{float(v):.{prec}f}"
        except Exception:
            return default
    
    for r in data:
        print(
            f"{r.get('model','N/A'):<20} "
            f"{str(r.get('run','N/A')):<4} "
            f"{str(r.get('epoch','N/A')):<6} "
            f"{_f(r,'AP_video_lit',4):<10} "
            f"{_f(r,'AUC_video_lit',4):<10} "
            f"{_f(r,'mTTA_free_lit',2):<10} "
            f"{_f(r,'TTA_R80',2):<10}"
        )
    
    print("\n📝 EPOCH: época del checkpoint con mejor AP_video_lit")
    print("📝 Todas las métricas son sobre el conjunto de validación\n")

def main():
    """Loop principal: entrena todos los modelos con N repeticiones."""
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*90)
    print("🎯 BATCH TRAINING PIPELINE")
    print("="*90)
    print(f"📁 Experimentos: {EXPERIMENTS_ROOT}")
    print(f"🔢 Modelos: {len(MODELS)}")
    print(f"🔁 Repeticiones por modelo: {REPEATS}")
    print(f"📊 Protocolo: {AP_SCOPE.upper()} (comparable con CRASH/LATTE)")
    print("="*90 + "\n")
    
    for run_idx in range(START_RUN_INDEX, START_RUN_INDEX + REPEATS):
        print(f"\n{'='*90}")
        print(f"🔄 RUN {run_idx + 1}/{START_RUN_INDEX + REPEATS}")
        print(f"{'='*90}")
        for md in MODELS:
            _ = run_one(md, run_idx)

    print("\n" + "="*90)
    print("📈 Recopilando resultados finales...")
    print("="*90)
    
    rows = []
    for run_idx in range(START_RUN_INDEX, START_RUN_INDEX + REPEATS):
        for md in MODELS:
            found = False
            for exp_dir in candidate_dirs_for(md, run_idx):
                summary = load_summary_for(exp_dir)
                if summary is not None:
                    summary["model"] = md
                    summary["run"] = run_idx
                    rows.append(summary)
                    found = True
                    break
            
            if not found and run_idx == START_RUN_INDEX:
                expected = preferred_dir_name(md)
                print(
                    f"⚠️  [WARN] No checkpoint para {md} run={run_idx}. "
                    f"Esperado en: {expected}_{run_idx}"
                )
    
    print_table(rows)
    
    if rows:
        import json as _json
        summary_path = EXPERIMENTS_ROOT / f"summary_{AP_SCOPE}.json"
        with summary_path.open("w") as f:
            _json.dump(rows, f, indent=2)
        print(f"💾 Resumen guardado en: {summary_path}\n")

if __name__ == "__main__":
    main()