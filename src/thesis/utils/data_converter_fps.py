#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convierte vídeos (p.ej., DAD 4s@25fps) a 5.0 s @ 20 fps manteniendo ~el MISMO tamaño.
- Estira el tiempo: setpts=1.25*PTS (4.0s → 5.0s)
- Re-muestrea fps: fps=20
- Re-encode 2-PASS x264 con bitrate ≈ bitrate_origen * factor (por defecto 1.05)
- Sin audio (-an)
- Salta si ya está ≈5.0s y ≈20fps
"""

from __future__ import annotations
import argparse, concurrent.futures as fx, os, shutil, subprocess, tempfile
from pathlib import Path
from typing import Tuple

VIDEO_EXTS = (".mp4",".mkv",".avi",".mov",".MP4",".MOV")

def run(cmd:list[str])->str:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode!=0: raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()

def probe_num(path:str, entries:str)->float:
    """
    ffprobe helper (devuelve float; si 'a/b' → a/b).
    - entries ejemplo:
      * 'format=duration'
      * 'stream=avg_frame_rate'
      * 'stream=bit_rate'
    """
    try:
        out = run([
            "ffprobe","-v","error",
            "-select_streams","v:0",
            "-show_entries", entries,
            "-of","default=nw=1:nk=1",
            path
        ]).splitlines()[0].strip()
        if "/" in out:
            a,b = out.split("/",1)
            return float(a)/max(float(b),1.0)
        return float(out)
    except Exception:
        return 0.0

def probe_duration(path:str)->float:
    try:
        out = run([
            "ffprobe","-v","error",
            "-show_entries","format=duration",
            "-of","default=nw=1:nk=1",
            path
        ]).splitlines()[0].strip()
        return float(out)
    except Exception:
        return 0.0

def info(path:str)->Tuple[float,float]:
    dur = probe_duration(path)
    fps = probe_num(path, "stream=avg_frame_rate")
    return dur, fps

def needs(dur:float,fps:float)->bool:
    return not (abs(dur-5.0)<=0.1 and abs(fps-20.0)<=0.2)

def get_src_bitrate_bps(path:Path)->int:
    """
    Intenta leer el bitrate del stream de vídeo (bps).
    Si no existe, lo estima con size/duration (incluye audio si lo hubiera).
    """
    br = int(probe_num(str(path), "stream=bit_rate"))
    if br > 0:
        return br
    # Fallback por tamaño/tiempo
    dur = max(probe_duration(str(path)), 1e-6)
    size_bytes = path.stat().st_size
    est = int((size_bytes * 8) / dur)  # bps (incluye audio si existiera)
    # Como quitamos audio (-an), podemos aplicar un factor reductor ligero
    # para no sobredimensionar en exceso. Lo dejamos tal cual y delegamos al
    # factor del usuario (--bitrate-factor).
    return max(est, 100_000)  # ≥100 kbps por seguridad

def two_pass_encode_match_bitrate(
    src:Path, dst:Path, target_bps:int, preset:str, passlogfile:Path
)->None:
    """
    2-pass x264 a bitrate objetivo (ABR), sin audio, 5.0s@20fps.
    """
    vf = "setpts=1.25*PTS,fps=20"
    # 1ª pasada → /dev/null
    cmd1 = [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i", str(src),
        "-map","0:v:0",
        "-vf", vf,
        "-an",
        "-c:v","libx264",
        "-preset", preset,
        "-b:v", str(target_bps),
        "-maxrate", str(target_bps),
        "-bufsize", str(target_bps*2),
        "-pass","1",
        "-passlogfile", str(passlogfile),
        "-f","mp4","/dev/null"
    ]
    run(cmd1)

    # 2ª pasada → dst
    cmd2 = [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i", str(src),
        "-map","0:v:0",
        "-vf", vf,
        "-an",
        "-c:v","libx264",
        "-preset", preset,
        "-b:v", str(target_bps),
        "-maxrate", str(target_bps),
        "-bufsize", str(target_bps*2),
        "-pass","2",
        "-passlogfile", str(passlogfile),
        "-pix_fmt","yuv420p",
        "-movflags","+faststart",
        str(dst)
    ]
    run(cmd2)

def convert_one(
    src:Path, dst:Path, preset:str, overwrite:bool, factor:float
)->str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return "exists"

    d,f = info(str(src))
    if not needs(d,f):
        if not dst.exists():
            shutil.copy2(src,dst)
        return f"skip (dur≈{d:.2f}s,fps≈{f:.2f})"

    src_bps = get_src_bitrate_bps(src)
    tgt_bps = int(src_bps * factor)

    # Log temporal por clip (evita colisiones en multihilo)
    with tempfile.TemporaryDirectory(prefix="ffmpeg_pass_") as tmpd:
        passlog = Path(tmpd) / "x264_2pass"
        two_pass_encode_match_bitrate(src, dst, tgt_bps, preset, passlog)

    # Informe post
    d2,f2 = info(str(dst))
    out_size = dst.stat().st_size if dst.exists() else 0
    in_size  = src.stat().st_size
    ratio = (out_size / max(in_size,1)) if in_size>0 else 0.0
    return (f"done (src_bitrate≈{src_bps/1e6:.2f}Mbps → tgt≈{tgt_bps/1e6:.2f}Mbps; "
            f"dur≈{d2:.2f}s,fps≈{f2:.2f}; size×{ratio:.2f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--preset", default="slow",
                    help="x264 preset (ultrafast..placebo). Recomendado: slow/medium.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--bitrate-factor", type=float, default=1.05,
                    help="Multiplicador sobre el bitrate de origen (p.ej. 1.05 = +5%).")
    a = ap.parse_args()

    srcs = [p for p in Path(a.in_root).rglob("*") if p.suffix in VIDEO_EXTS]
    print(f"[INFO] vídeos: {len(srcs)}")

    def job(p:Path):
        rel = p.relative_to(a.in_root)
        dst = (Path(a.out_root)/rel).with_suffix(".mp4")
        try:
            msg = convert_one(p, dst, a.preset, a.overwrite, a.bitrate_factor)
        except Exception as e:
            msg = f"ERR {e}"
        print(f"{p} -> {msg}")

    with fx.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(job, srcs))

    print("[DONE]")

if __name__ == "__main__":
    main()
