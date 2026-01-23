#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import torch
from collections import OrderedDict
from transformers import AutoModel

ENC_PREFIX_CANDIDATES = [
    "encoder.",
    "model.encoder.",
    "backbone.",
    "module.encoder.",
    "module.backbone.",
]

def _maybe_get_sd(obj):
    # Devuelve un state_dict dict-like
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "module", "net"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    return obj

def _find_encoder_prefix(sd):
    keys = list(sd.keys())
    for pre in ENC_PREFIX_CANDIDATES:
        has_vit = any(
            k.startswith(pre + p)
            for p in ("pos_embed", "blocks.", "patch_embed.", "cls_token")
            for k in keys
        )
        if has_vit:
            return pre
    # fallback: sin prefijo
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--badas_pth", required=True,
                    help="Checkpoint BADAS original (badas_open.pth)")
    ap.add_argument("--hf_repo", required=True,
                    help="HF repo V-JEPA2, ej: facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--out", required=True,
                    help="Salida: checkpoint BADAS con encoder HF inyectado")
    args = ap.parse_args()

    print(f"[LOAD] HF encoder from: {args.hf_repo}")
    hf_model = AutoModel.from_pretrained(args.hf_repo, trust_remote_code=True)
    hf_sd = hf_model.state_dict()

    print(f"[LOAD] BADAS checkpoint from: {args.badas_pth}")
    raw = torch.load(args.badas_pth, map_location="cpu")
    badas_sd = _maybe_get_sd(raw)

    enc_prefix = _find_encoder_prefix(badas_sd)
    print(f"[INFO] detected BADAS encoder prefix: '{enc_prefix}'")

    updated = 0
    skipped_shape = 0

    for k, v in hf_sd.items():
        badas_k = enc_prefix + k
        if badas_k in badas_sd:
            if tuple(badas_sd[badas_k].shape) == tuple(v.shape):
                badas_sd[badas_k] = v
                updated += 1
            else:
                skipped_shape += 1

    print(f"[INFO] encoder params updated: {updated}")
    print(f"[INFO] shape mismatches      : {skipped_shape}")

    # Volcar de nuevo el checkpoint en formato original
    if isinstance(raw, dict):
        # Si el raw tenía una subestructura 'state_dict', respetarla
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw["state_dict"].update(badas_sd)
        else:
            raw.update(badas_sd)
        torch.save(raw, args.out)
    else:
        torch.save(badas_sd, args.out)

    print(f"[OK] Saved BADAS with HF encoder to: {args.out}")

if __name__ == "__main__":
    main()
