#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# ==== Imports de tu repo ====
from models import build_model
from dataset_embeddings import VJEPAFramewiseDataset, collate_pad
from dataset_tokens import DatasetTokens, collate_tokens
from metrics import (
    compute_val_metrics_from_frame_logits,
    mtta_threshold_free_lit,
    tta_at_threshold,
)

# -----------------------------
# Utils
# -----------------------------
def _iter_pts(dirs: List[str]):
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        yield from p.glob("*.pt")

def _infer_embed_dim_and_tokens(sample_pt: Path, use_tokens_flag: Optional[bool]) -> Tuple[int, bool, Dict[str, Any]]:
    data = torch.load(sample_pt, map_location="cpu")
    geom = {}
    if use_tokens_flag is True or (use_tokens_flag is None and "tokens" in data):
        if "tokens" not in data:
            raise RuntimeError(f"{sample_pt} no contiene 'tokens'")
        D = int(data["tokens"].shape[-1])
        meta = data.get("tokens_meta", {}) or {}
        geom = {
            "grid_h": int(meta.get("grid_h", 16)),
            "grid_w": int(meta.get("grid_w", 16)),
            "frames_per_clip": int(meta.get("frames_per_clip", 16)),
            "tubelet_size": int(meta.get("tubelet_size", 2)),
        }
        return D, True, geom
    else:
        if "z_clip" not in data:
            raise RuntimeError(f"{sample_pt} no contiene 'z_clip'")
        D = int(data["z_clip"].shape[-1])
        return D, False, geom

def _build_model_from_ckpt(cfg: Dict[str, Any], embed_dim: int, n_windows: int):
    """Filtra cfg para evitar pasar duplicados/ruido a build_model()."""
    model_type = str(cfg.get("model_type"))

    blacklist = {
        "model_type", "embed_dim", "n_windows", "n_params",
        "ap_scope", "mode", "decision_point", "event_align",
        "use_tokens", "horizons", "window_s", "post_window_s",
        "dt_shift",
    }
    whitelist = {
        # Geometría tokens
        "grid_h", "grid_w", "frames_per_clip", "tubelet_size",
        # BADAS
        "M", "d", "n_heads", "num_attn_layers", "dropout", "attn_dropout",
        "upsample_to_frames",
        # TemporalAttentionHead
        "n_heads_spatial", "n_heads_temporal", "tfm_layers",
        "token_dropout_p", "temporal_dropout_p", "freeze_pooler", "proj_dim",
        # Embeddings heads (GRU/TCN/Transformer)
        "channels", "levels", "k",
        "d_model", "num_layers", "ff_dim", "pe_dropout", "pe_type", "attn_dropout",
        "max_len", "causal",
        # Deformable Event head
        "n_queries", "num_points", "num_levels", "d_model",
        "use_checkpoint", "align_corners", "delta_lambda",
    }
    kw = {k: v for k, v in cfg.items() if (k not in blacklist and k in whitelist)}
    return build_model(model_type=model_type, embed_dim=embed_dim, n_windows=n_windows, **kw)
def _forward_on_loader(model, ld: DataLoader, device: torch.device, use_tokens: bool):
    """Hace forward en TEST y paddea todos los lotes a T_max antes de concatenar."""
    logits_batches, masks_batches, deltas_batches, times_batches = [], [], [], []
    y_video_all, t_event_all = [], []

    model.eval()
    with torch.no_grad():
        for b in tqdm(ld, desc="TEST infer", leave=False):
            m = b["masks"].to(device)              # (B, T_b)
            d = b["deltas"].to(device)             # (B, T_b)
            t = b["clip_times"].to(device)         # (B, T_b)
            y_vid = b["targets"].cpu().numpy()     # (B,)
            t_evt = b["t_events"].cpu().numpy()    # (B,)

            if use_tokens:
                x_tok = b["tokens"].to(device)     # (B, T_b, L, D)
                tok_mask = b.get("token_mask", None)
                if tok_mask is not None:
                    tok_mask = tok_mask.to(device)  # (B, T_b, L)

                B, T_b, L, _ = x_tok.shape
                logits_bt = []
                for t_clip in range(T_b):
                    clip_tokens = x_tok[:, t_clip, :, :]                      # (B, L, D)
                    clip_mask   = tok_mask[:, t_clip, :] if tok_mask is not None else None
                    logits_clip = model(clip_tokens, token_mask=clip_mask)    # (B, Dt?, nW) o (B, nW)
                    logits_t = logits_clip[:, -1, :] if logits_clip.ndim == 3 else logits_clip
                    logits_bt.append(logits_t)                                 # (B, nW)
                logits_b = torch.stack(logits_bt, dim=1)                       # (B, T_b, nW)
            else:
                x_emb = b["embeddings"].to(device)  # (B, T_b, D)
                logits_b = model(x_emb, mask=m)     # (B, T_b, nW)

            logits_batches.append(logits_b.cpu())
            masks_batches.append(m.cpu())
            deltas_batches.append(d.cpu())
            times_batches.append(t.cpu())
            y_video_all.extend(y_vid.tolist())
            t_event_all.extend(t_evt.tolist())

    # === Pad a T_max y concat ===
    T_max = max(x.shape[1] for x in logits_batches)
    nW = logits_batches[0].shape[-1]

    def _pad2T(x, T_max, pad_val=0.0):
        # x: (B, T, ...)   → pad en dim=1
        B, T = x.shape[0], x.shape[1]
        if T == T_max:
            return x
        pad_sizes = (0, 0)  # por defecto sin última dim
        if x.dim() == 3:    # (B, T, nW)
            pad_sizes = (0, 0, 0, T_max - T)  # pad (nW:0, T:delta)
            return torch.nn.functional.pad(x, pad_sizes, value=pad_val)
        elif x.dim() == 2:  # (B, T)
            pad_sizes = (0, T_max - T)
            return torch.nn.functional.pad(x, pad_sizes, value=pad_val)
        else:
            raise RuntimeError(f"Dim no soportada para pad: {x.shape}")

    logits_batches = [_pad2T(x, T_max, pad_val=0.0) for x in logits_batches]          # (B, T_max, nW)
    masks_batches  = [_pad2T(x, T_max, pad_val=0.0) for x in masks_batches]           # (B, T_max)  [bool luego]
    deltas_batches = [_pad2T(x, T_max, pad_val=0.0) for x in deltas_batches]          # (B, T_max)
    times_batches  = [_pad2T(x, T_max, pad_val=0.0) for x in times_batches]           # (B, T_max)

    logits_bt_w = torch.cat(logits_batches, dim=0)             # (N, T_max, nW)
    mask_bt     = torch.cat(masks_batches,  dim=0).bool()      # (N, T_max)
    deltas_bt   = torch.cat(deltas_batches, dim=0)             # (N, T_max)
    times_bt    = torch.cat(times_batches,  dim=0)             # (N, T_max)

    y_video_np  = np.asarray(y_video_all)                      # (N,)
    t_events_np = np.asarray(t_event_all)                      # (N,)

    # Sanity básico
    assert logits_bt_w.shape[0] == y_video_np.shape[0], "Mismatch N vídeos"
    assert (mask_bt.sum(dim=1) > 0).all(), "Hay vídeos sin frames válidos"

    return logits_bt_w, mask_bt, deltas_bt, times_bt, y_video_np, t_events_np


def _find_experiments(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if (p / "best_model.pt").exists()])

def _default_test_dirs(dataset: str) -> List[str]:
    dataset = dataset.lower()
    if dataset == "ccd":
        return [
            "data_exp/vjepa2_tokens/ccd/test/testing/positive",
            "data_exp/vjepa2_tokens/ccd/test/testing/negative",
        ]
    if dataset == "dad":
        return [
            "data_exp/vjepa2_tokens/dad/test/testing/positive",
            "data_exp/vjepa2_tokens/dad/test/testing/negative",
        ]
    raise ValueError("dataset desconocido. Usa --test-dirs para modo custom.")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Evaluación en TEST (ANYTIME + PRE, sin leakage)")
    ap.add_argument("--dataset", type=str, choices=["ccd", "dad"], default=None)
    ap.add_argument("--test-dirs", nargs="+", default=None)
    ap.add_argument("--experiments-root", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    exp_root = Path(args.experiments_root).resolve()
    if not exp_root.exists():
        sys.exit(f"{exp_root} no existe")

    if args.test_dirs is None:
        assert args.dataset is not None, "Proporciona --dataset o --test-dirs"
        test_dirs = _default_test_dirs(args.dataset)
        print(f"[INFO] Dataset='{args.dataset.upper()}' TEST:")
        for d in test_dirs: print(f"  - {d}")
    else:
        test_dirs = args.test_dirs
        print("[INFO] TEST dirs:")
        for d in test_dirs: print(f"  - {d}")
    print(f"[INFO] Experiments root: {exp_root}")

    exps = _find_experiments(exp_root)
    print(f"[INFO] Checkpoints a evaluar: {len(exps)}")
    if not exps:
        print("❌ No hay checkpoints en esa ruta.")
        sys.exit(1)

    if not len(list(_iter_pts(test_dirs))):
        sys.exit("❌ TEST dirs vacíos o extensión incorrecta.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for exp in exps:
        ckpt_path = exp / "best_model.pt"
        try:
            ck = torch.load(ckpt_path, map_location="cpu")
            cfg: Dict[str, Any] = dict(ck.get("config", {}))
            valm: Dict[str, Any] = dict(ck.get("val_metrics", {}))

            # --- config base coherente con train ---
            model_type = str(cfg.get("model_type"))
            horizons   = list(cfg.get("horizons", [5.0]))
            window_s   = float(cfg.get("window_s", 5.0))
            post_window_s = float(cfg.get("post_window_s", 0.0))
            dt_shift   = float(cfg.get("dt_shift", 0.0))
            use_tokens_flag = bool(cfg.get("use_tokens", False))

            # Inferir D y geometría real desde un sample de TEST
            first_pt = next(_iter_pts(test_dirs))
            D, is_token_data, geom_from_data = _infer_embed_dim_and_tokens(first_pt, use_tokens_flag)
            use_tokens = is_token_data
            geom = dict(geom_from_data)

            # Dataset + Loader  (❗ sin 'horizons=')
            if use_tokens:
                ds = DatasetTokens(test_dirs, window_s=window_s, post_window_s=post_window_s, video_ids=None)
                collate_fn = collate_tokens
                bs = args.batch_size or 1
            else:
                ds = VJEPAFramewiseDataset(test_dirs, window_s=window_s, post_window_s=post_window_s, video_ids=None)
                collate_fn = collate_pad
                bs = args.batch_size or 32

            ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)

            # Reconstruir modelo
            recon_cfg = dict(cfg); recon_cfg.update(geom)
            model = _build_model_from_ckpt(recon_cfg, embed_dim=D, n_windows=len(horizons)).to(device)

            # Forward en TEST
            logits_bt_w, mask_bt, deltas_bt, times_bt, y_video_np, t_events_np = _forward_on_loader(model, ld, device, use_tokens)
            probs_bt_w = torch.sigmoid(logits_bt_w)

            # Métricas ANYTIME
            valm_any = compute_val_metrics_from_frame_logits(
                logits_bt_w=logits_bt_w, mask_bt=mask_bt, deltas_bt=deltas_bt,
                y_video_t=torch.tensor(y_video_np, dtype=torch.float32),
                horizons=horizons, dt_shift=dt_shift, ap_scope="anytime",
                times_bt=times_bt, t_events=torch.tensor(t_events_np, dtype=times_bt.dtype),
            )
            # Métricas PRE
            valm_pre = compute_val_metrics_from_frame_logits(
                logits_bt_w=logits_bt_w, mask_bt=mask_bt, deltas_bt=deltas_bt,
                y_video_t=torch.tensor(y_video_np, dtype=torch.float32),
                horizons=horizons, dt_shift=dt_shift, ap_scope="pre",
                times_bt=times_bt, t_events=torch.tensor(t_events_np, dtype=times_bt.dtype),
            )

            # mTTA (threshold-free, PRE)
            mtta_free_pre = mtta_threshold_free_lit(
                probs_bt_w=probs_bt_w, times_bt=times_bt, mask_bt=mask_bt,
                t_events=t_events_np, y_video=y_video_np, q_grid="paper19",
                strict_no_cross=True, deltas_bt=deltas_bt, dt_shift=dt_shift,
            )

            # TTA@R80 con umbrales de VALIDACIÓN (no recalcular en TEST)
            thr_any_val = None
            thr_pre_val = None
            try:
                thr_any_val = float(valm.get("extras", {}).get("anytime", {}).get("thr_R80_lit"))
                thr_pre_val = float(valm.get("extras", {}).get("pre", {}).get("thr_R80_lit"))
            except Exception:
                pass
            if thr_any_val is None and "thr_R80_lit" in valm:
                try: thr_any_val = float(valm["thr_R80_lit"])
                except Exception: pass

            if thr_pre_val is not None:
                tta_r80_pre = tta_at_threshold(
                    probs_bt_w=probs_bt_w, times_bt=times_bt, mask_bt=mask_bt,
                    t_events=t_events_np, thr=thr_pre_val, y_video=y_video_np,
                    strict_no_cross=True, deltas_bt=deltas_bt, dt_shift=dt_shift,
                )
            else:
                tta_r80_pre = None

            if thr_any_val is not None:
                tta_r80_any = tta_at_threshold(
                    probs_bt_w=probs_bt_w, times_bt=times_bt, mask_bt=mask_bt,
                    t_events=t_events_np, thr=thr_any_val, y_video=y_video_np,
                    strict_no_cross=True, deltas_bt=deltas_bt, dt_shift=dt_shift,
                )
            else:
                tta_r80_any = None

            # Guardar predicciones crudas
            torch.save({
                "logits_bt_w": logits_bt_w, "probs_bt_w": probs_bt_w, "mask_bt": mask_bt,
                "deltas_bt": deltas_bt, "times_bt": times_bt,
                "y_video": y_video_np, "t_events": t_events_np,
                "thr_any_val": thr_any_val, "thr_pre_val": thr_pre_val,
                "window_s": window_s, "horizons": horizons, "use_tokens": use_tokens, "dt_shift": dt_shift,
            }, exp / "test_predictions.pt")

            row = {
                "exp": exp.name,
                "model": model_type,
                "AP_any": float(valm_any.get("AP_video_lit", 0.0)),
                "AUC_any": float(valm_any.get("AUC_video_lit", 0.5)),
                "AP_pre": float(valm_pre.get("AP_video_lit", 0.0)),
                "AUC_pre": float(valm_pre.get("AUC_video_lit", 0.5)),
                "mTTA_free_pre": float(mtta_free_pre) if np.isfinite(mtta_free_pre) else None,
                "TTA_R80_pre": float(tta_r80_pre) if (tta_r80_pre is not None and np.isfinite(tta_r80_pre)) else None,
                "TTA_R80_any": float(tta_r80_any) if (tta_r80_any is not None and np.isfinite(tta_r80_any)) else None,
                "thr_any_val": thr_any_val, "thr_pre_val": thr_pre_val,
            }
            rows.append(row)

            print(f"[OK] {exp.name} | ANY(AP={row['AP_any']:.4f},AUC={row['AUC_any']:.4f}) "
                  f"| PRE(AP={row['AP_pre']:.4f},AUC={row['AUC_pre']:.4f}, mTTA={row['mTTA_free_pre']:.2f}, "
                  f"TTA@R80_pre={row['TTA_R80_pre']})")

        except Exception as e:
            print(f"[SKIP] {exp.name}: {e}")

    if not rows:
        print("\n❌ No hay resultados.")
        return

    rows.sort(key=lambda r: r["AP_any"], reverse=True)
    print("\n" + "="*95)
    print("📊 TEST SUMMARY (ANYTIME + PRE; sin leakage)")
    print("="*95)
    hdr = f"{'EXP':<45} {'AP_any':<8} {'AUC_any':<8} {'AP_pre':<8} {'AUC_pre':<8} {'mTTA_pre':<9} {'TTA@R80_pre':<12} {'TTA@R80_any':<12}"
    print(hdr)
    print("-"*95)
    def _fmt(x, n=4):
        if x is None: return "None"
        try: return f"{float(x):.{n}f}"
        except: return "None"
    for r in rows:
        print(f"{r['exp']:<45} "
              f"{_fmt(r['AP_any']):<8} {_fmt(r['AUC_any']):<8} "
              f"{_fmt(r['AP_pre']):<8} {_fmt(r['AUC_pre']):<8} "
              f"{_fmt(r['mTTA_free_pre'],2):<9} "
              f"{_fmt(r['TTA_R80_pre'],2):<12} {_fmt(r['TTA_R80_any'],2):<12}")

    out_json = exp_root / "summary_test_pre_any.json"
    with out_json.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n💾 Guardado: {out_json}")

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_DISABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
