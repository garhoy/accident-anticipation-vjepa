#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_badas_encoder_hf.py

Extrae el encoder ViT de un checkpoint BADAS-Open y lo adapta al
formato de un modelo HuggingFace (por ejemplo V-JEPA2 ViT-L).

NOVEDAD:
  - El argumento --badas-pth puede ser:
      * una ruta local a un .pth
      * O un repo_id de HuggingFace (ej: nexar-ai/BADAS-Open)

Si la ruta no existe en disco, se asume que es un repo HF y se descarga
'weights/badas_open.pth' con huggingface_hub.hf_hub_download.

Ejemplo de uso (vía HuggingFace):

  CUDA_VISIBLE_DEVICES=0 python examples/extract_badas_encoder_hf.py \\
    --badas-pth nexar-ai/BADAS-Open \\
    --hf-repo facebook/vjepa2-vitl-fpc16-256-ssv2 \\
    --out /home/ander/BADAS-Open/weights/badas_encoder_vitL_hf.pt
"""

import argparse
import os
import re
from collections import OrderedDict

import torch
from transformers import AutoModel


def _maybe_get_sd(obj):
    """
    Intenta extraer un state_dict (dict de tensores) de distintos formatos típicos:
    - dict con claves 'model', 'state_dict', 'module', 'net'
    - dict plano de parámetros
    """
    if isinstance(obj, dict):
        for k in ["model", "state_dict", "module", "net"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    return obj


def _strip_encoder_prefix(sd):
    """
    Devuelve un sub-dict que corresponde al encoder (ViT) quitando prefijos
    típicos como 'encoder.', 'model.encoder.', 'module.encoder.', 'backbone.', etc.
    Si no encuentra un prefijo claro, devuelve el dict original (se filtrará luego).
    """
    keys = list(sd.keys())
    if not keys:
        return sd

    candidate_prefixes = [
        "encoder.",
        "model.encoder.",
        "module.encoder.",
        "backbone.",
        "model.backbone.",
        "module.backbone.",
    ]

    for pre in candidate_prefixes:
        if any(k.startswith(pre) for k in keys):
            return OrderedDict(
                (re.sub(r"^" + re.escape(pre), "", k), v)
                for k, v in sd.items()
                if k.startswith(pre)
            )

    # Si ya parece un ViT plano (pos_embed, blocks, patch_embed, cls_token, etc.)
    if any(
        k.startswith(("pos_embed", "blocks", "patch_embed", "cls_token"))
        for k in keys
    ):
        return sd

    # Si no sabemos, devolvemos tal cual y luego intersecamos con HF por nombre+shape
    return sd


def _intersect_by_name_and_shape(src_sd, tgt_sd):
    """
    Devuelve un dict con las claves que existen en ambos state_dict y
    tienen la misma shape. Ignora todo lo demás.
    """
    matched = {}
    miss_shape = 0
    for k, v in src_sd.items():
        if k in tgt_sd:
            if tuple(v.shape) == tuple(tgt_sd[k].shape):
                matched[k] = v
            else:
                miss_shape += 1
    return matched, miss_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--badas-pth",
        required=True,
        help=(
            "Ruta al checkpoint BADAS-Open (.pth) "
            "O repo_id de HuggingFace (ej: nexar-ai/BADAS-Open)"
        ),
    )
    parser.add_argument(
        "--hf-repo",
        required=True,
        help="Repo HF objetivo (ej: facebook/vjepa2-vitl-fpc16-256-ssv2)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Ruta de salida .pt con SOLO encoder en layout HF",
    )
    args = parser.parse_args()

    # 1. Cargar modelo HF de referencia
    print(f"[LOAD] Modelo HF objetivo: {args.hf_repo}")
    hf_model = AutoModel.from_pretrained(args.hf_repo, trust_remote_code=True)
    tgt_sd = hf_model.state_dict()
    print(f"[INFO] Parámetros en modelo HF: {len(tgt_sd)} tensores")

    # 2. Resolver checkpoint BADAS: local o HuggingFace
    if os.path.isfile(args.badas_pth):
        ckpt_path = args.badas_pth
        print(f"[LOAD] Checkpoint BADAS local: {ckpt_path}")
    else:
        # Interpretamos badas-pth como repo_id de HF
        repo_id = args.badas_pth
        print(
            f"[INFO] '{args.badas_pth}' no existe como archivo. "
            f"Se interpreta como repo HF: {repo_id}"
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print(
                "[ERR] huggingface_hub no está instalado y no se ha encontrado "
                "un .pth local. Instala con: pip install huggingface_hub"
            )
            return

        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename="weights/badas_open.pth",
        )
        print(f"[LOAD] Checkpoint BADAS descargado desde HF: {ckpt_path}")

    # 3. Cargar checkpoint BADAS
    print(f"[LOAD] Abriendo checkpoint BADAS: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")

    src_sd_full = _maybe_get_sd(raw)
    print(
        f"[INFO] Tensores totales en checkpoint BADAS (antes de filtrar): "
        f"{len(src_sd_full)}"
    )

    src_sd_enc = _strip_encoder_prefix(src_sd_full)
    print(
        f"[INFO] Tensores candidatos del encoder BADAS (tras quitar prefijos): "
        f"{len(src_sd_enc)}"
    )

    # 4. Intersección por nombre+shape
    matched, miss_shape = _intersect_by_name_and_shape(src_sd_enc, tgt_sd)
    print(f"[MATCH] Emparejados nombre+shape     : {len(matched)}")
    print(f"[MATCH] Descartes por mismatch shape : {miss_shape}")

    if len(matched) == 0:
        print(
            "[WARN] No se ha emparejado NINGÚN tensor entre BADAS y HF. "
            "Probable incompatibilidad de nombres o arquitectura."
        )
        return

    # 5. Guardar solo los pesos compatibles con HF
    torch.save(matched, args.out)
    print(f"[OK] Guardado encoder BADAS->HF en: {args.out}")

    # Info extra sobre pos_embed
    pe = matched.get("pos_embed", None)
    if pe is not None and pe.ndim == 2:
        n_tokens = pe.shape[1]
        steps = (n_tokens - 1) / 256.0
        clip_len = int(round(2 * steps))
        print(
            f"[HINT] pos_embed tokens={n_tokens} ⇒ steps≈{steps:.2f}, "
            f"clip_len≈{clip_len} (usa repo HF con fpc{clip_len} si quieres ser estricto)"
        )


if __name__ == "__main__":
    main()
