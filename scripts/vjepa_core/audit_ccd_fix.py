#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import torch
import sys

def iter_pts(d):
    p = Path(d)
    if not p.exists():
        return
    yield from p.glob("*.pt")

def read_target(pt):
    """Lee el target del .pt y el objeto entero."""
    try:
        o = torch.load(pt, map_location="cpu")
        meta = o.get("meta") or {}
        try:
            t = int(meta.get("target", -1))
        except Exception:
            t = -1
        return t, o
    except Exception as e:
        print(f"[ERROR] No se pudo leer {pt}: {e}", file=sys.stderr)
        return -99, None # Retorna un objeto None para skipear

def set_target_and_save(pt, obj, new_t: int):
    """Sobreescribe el meta.target y guarda el archivo."""
    obj.setdefault("meta", {})["target"] = int(new_t)
    torch.save(obj, pt)

def main():
    ap = argparse.ArgumentParser("Audita y repara meta.target en CCD test")
    ap.add_argument("--pos-dir", required=True, help=".../ccd/test/testing/positive")
    ap.add_argument("--neg-dir", required=True, help=".../ccd/test/testing/negative")
    ap.add_argument("--fix-pt", dest="fix_pt", action="store_true",
                    help="Reescribe meta.target para que coincida con la carpeta")
    args = ap.parse_args()

    # Versión CORREGIDA
    pos = list(iter_pts(args.pos_dir))
    neg = list(iter_pts(args.neg_dir))
    print(f"[INFO] Archivos: pos={len(pos)}  neg={len(neg)}  total={len(pos)+len(neg)}")
    
    if not pos and not neg:
        print("[ERROR] No se encontraron archivos .pt en las rutas. Revisa las --pos-dir y --neg-dir.", file=sys.stderr)
        sys.exit(1)

    mism = []

    # Positivos: deben tener target=1
    print("\nAuditando positivos...")
    for pt in pos:
        t, o = read_target(pt)
        if o is None: continue # Error al leer
        if t != 1:
            mism.append(("POS→meta!=1", pt, o, 1)) # (tag, path, objeto, target_correcto)

    # Negativos: deben tener target=0
    print("Auditando negativos...")
    for pt in neg:
        t, o = read_target(pt)
        if o is None: continue # Error al leer
        if t != 0:
            mism.append(("NEG→meta!=0", pt, o, 0)) # (tag, path, objeto, target_correcto)

    if not mism:
        print("\n[OK] No hay desajustes meta.target vs carpeta.")
    else:
        print(f"\n[WARN] Desajustes encontrados: {len(mism)}")
        for tag, pt, _, _ in mism[:30]:
            print(f"  - {tag}: {pt.name}")
        
        if args.fix_pt:
            print("\n[FIX] Aplicando correcciones...")
            for tag, pt, o, new_t in mism:
                try:
                    set_target_and_save(pt, o, new_t)
                except Exception as e:
                    print(f"[ERROR] No se pudo guardar {pt}: {e}", file=sys.stderr)
            
            # Verificación rápida tras el fix
            print("[FIX] Verificando correcciones...")
            pos_bad = sum(read_target(pt)[0] != 1 for pt in pos)
            neg_bad = sum(read_target(pt)[0] != 0 for pt in neg)
            print(f"[FIX] Resumen post-fix: pos_mal={pos_bad}  neg_mal={neg_bad}")
            if pos_bad == 0 and neg_bad == 0:
                print("[OK] Etiquetas corregidas.")
            else:
                print("[WARN] Aún quedan etiquetas incorrectas tras el fix.")

if __name__ == "__main__":
    main()