#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, torch
from collections import OrderedDict
from transformers import AutoModel

def _maybe_get_sd(obj):
    # Devuelve un dict de tensores (state_dict-like)
    if isinstance(obj, dict):
        # casos comunes
        for k in ["model", "state_dict", "module", "net"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    return obj

def _strip_encoder_prefix(sd):
    # Extrae sub-dict del encoder si viene con prefijos típicos
    keys = list(sd.keys())
    for pre in ["encoder.", "model.encoder.", "module.encoder.", "backbone.", "model.backbone."]:
        if any(k.startswith(pre) for k in keys):
            return OrderedDict((re.sub("^"+re.escape(pre), "", k), v)
                               for k,v in sd.items() if k.startswith(pre))
    # Si ya son pesos "planos" del ViT (pos_embed, blocks, patch_embed, etc.)
    if any(k.startswith(("pos_embed","blocks","patch_embed","cls_token")) for k in keys):
        return sd
    # Si no encontramos nada claro, devolvemos tal cual (se filtrará por intersección)
    return sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--badas_pth", required=True, help="weights/badas_open.pth (~4GB)")
    ap.add_argument("--hf_repo",   required=True, help="p.ej. facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--out",       required=True, help="salida .pt con SOLO encoder en layout HF")
    args = ap.parse_args()

    print(f"[LOAD] HF target model: {args.hf_repo}")
    hf_model = AutoModel.from_pretrained(args.hf_repo, trust_remote_code=True)
    tgt_sd = hf_model.state_dict()  # claves destino (encoder + posible head si existe)

    print(f"[LOAD] BADAS checkpoint: {args.badas_pth}")
    raw = torch.load(args.badas_pth, map_location="cpu")
    src_sd_full = _maybe_get_sd(raw)
    src_sd_enc  = _strip_encoder_prefix(src_sd_full)

    # Intersección por nombre y shape → robusto a diferencias de nombres de la head
    matched = {}
    miss_shape = 0
    for k, v in src_sd_enc.items():
        if k in tgt_sd:
            if tuple(v.shape) == tuple(tgt_sd[k].shape):
                matched[k] = v
            else:
                miss_shape += 1

    print(f"[INFO] tensores en BADAS (posibles): {len(src_sd_enc)}")
    print(f"[INFO] tensores en HF target      : {len(tgt_sd)}")
    print(f"[INFO] emparejados por nombre+shape: {len(matched)}  | shape_mismatch: {miss_shape}")

    # Guardamos SOLO el subset que casa con el modelo HF
    torch.save(matched, args.out)
    print(f"[OK] Guardado encoder en formato HF: {args.out}")

    # Heurística rápida del fpc a partir de pos_embed si existe
    pe = matched.get("pos_embed", None)
    if pe is not None and pe.ndim == 2:
        n_tokens = pe.shape[1]
        # V-JEPA2 suele: 1 (CLS) + steps * (H*W), con H=W=16 ⇒ H*W=256; tubelet=2 ⇒ steps=clip_len/2
        steps = (n_tokens - 1) / 256.0
        clip_len = int(round(2 * steps))
        print(f"[HINT] pos_embed tokens={n_tokens} ⇒ clip_len≈{clip_len}  (usa HF repo fpc{clip_len})")

if __name__ == "__main__":
    main()
