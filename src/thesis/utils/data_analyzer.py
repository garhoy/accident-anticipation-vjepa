
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Dict, Optional, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def has_ffprobe() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False

def parse_fraction(frac: Optional[str]) -> Optional[float]:
    if not frac or frac == "0/0":
        return None
    if "/" in frac:
        try:
            num, den = frac.split("/", 1)
            num = float(num); den = float(den)
            if den == 0:
                return None
            return num / den
        except Exception:
            return None
    try:
        return float(frac)
    except Exception:
        return None

def probe_ffprobe(path: Path) -> Dict[str, Optional[float]]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=index,codec_name,codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json",
        str(path)
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    t1 = time.perf_counter()
    if p.returncode != 0:
        return {"tool": "ffprobe", "fps": None, "n_frames": None, "w": None, "h": None, "duration": None, "codec": None, "probe_time_s": t1 - t0}
    try:
        data = json.loads(p.stdout)
    except Exception:
        return {"tool": "ffprobe", "fps": None, "n_frames": None, "w": None, "h": None, "duration": None, "codec": None, "probe_time_s": t1 - t0}
    duration = None
    if "format" in data and "duration" in data["format"]:
        try:
            duration = float(data["format"]["duration"])
        except Exception:
            duration = None
    fps = n_frames = w = h = None
    codec = None
    for s in data.get("streams", []):
        if s.get("codec_type") != "video":
            continue
        codec = s.get("codec_name") or codec
        w = int(s["width"]) if s.get("width") is not None else w
        h = int(s["height"]) if s.get("height") is not None else h
        fps = parse_fraction(s.get("avg_frame_rate")) or parse_fraction(s.get("r_frame_rate")) or fps
        try:
            n_frames = int(s.get("nb_frames")) if s.get("nb_frames") not in (None, "N/A") else None
        except Exception:
            n_frames = None
        break
    if n_frames is None and duration is not None and fps is not None:
        n_frames = int(round(duration * fps))
    return {
        "tool": "ffprobe",
        "fps": fps, "n_frames": n_frames, "w": w, "h": h, "duration": duration, "codec": codec,
        "probe_time_s": t1 - t0
    }

def probe_opencv(path: Path) -> Dict[str, Optional[float]]:
    try:
        import cv2
    except Exception as e:
        return {"tool": "opencv", "fps": None, "n_frames": None, "w": None, "h": None, "duration": None, "codec": None, "probe_time_s": None}
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"tool": "opencv", "fps": None, "n_frames": None, "w": None, "h": None, "duration": None, "codec": None, "probe_time_s": None}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (n_frames / fps) if (fps and n_frames) else None
        t1 = time.perf_counter()
        return {"tool": "opencv",
                "fps": float(fps) if fps else None,
                "n_frames": n_frames if n_frames else None,
                "w": w if w else None, "h": h if h else None,
                "duration": duration, "codec": None,
                "probe_time_s": t1 - t0}
    finally:
        cap.release()

def human(x, prec=3):
    if x is None:
        return "-"
    return f"{x:.{prec}f}" if isinstance(x, float) else str(x)

def main():
    import csv, statistics as st
    ap = argparse.ArgumentParser(description="Inspeccion de videos: FPS, frames, duracion y resolucion (recursivo).")
    ap.add_argument("root", type=str, help="Archivo de video o carpeta a escanear.")
    ap.add_argument("--csv-out", type=str, default=None, help="Guardar un CSV con los resultados.")
    ap.add_argument("--glob", type=str, default=None, help="Glob adicional (ej: *.mp4).")
    ap.add_argument("--show-n", type=int, default=50, help="Cuantos imprimir por consola (por si hay muchos).")
    ap.add_argument("--target-fps", type=float, default=20.0, help="FPS esperados (20.0 para DAD/A3D, 10.0 para CCD).")
    ap.add_argument("--target-frames", type=int, default=100, help="Frames esperados (100 para DAD/A3D, 50 para CCD).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] No existe: {root}", file=sys.stderr); sys.exit(2)

    if root.is_file():
        files = [root]
    else:
        if args.glob:
            files = list(root.rglob(args.glob))
        else:
            files = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]

    if not files:
        print("[INFO] No se encontraron videos.")
        sys.exit(0)

    rows = []
    for p in sorted(files):
        info = probe_ffprobe(p) if has_ffprobe() else probe_opencv(p)
        fps = info.get("fps")
        n_frames = info.get("n_frames")
        duration = info.get("duration")
        w = info.get("w"); h = info.get("h")
        status_bits = []
        if fps is not None and abs(fps - args.target_fps) <= 0.3: status_bits.append("fpsOK")
        if n_frames is not None and n_frames == args.target_frames: status_bits.append("NOK")
        if duration is not None and abs(duration - (args.target_frames/args.target_fps)) <= 0.25: status_bits.append("TOK")
        status = ",".join(status_bits) if status_bits else "check"

        rows.append({
            "path": str(p),
            "tool": info.get("tool"),
            "codec": info.get("codec"),
            "w": w, "h": h,
            "fps": fps,
            "n_frames": n_frames,
            "duration_s": duration,
            "status": status,
            "probe_time_s": info.get("probe_time_s"),
        })

    print("path,fps,n_frames,duration_s,resolution,status")
    for i, r in enumerate(rows[:args.show_n]):
        # ... dentro del bucle donde imprimes cada fila
        res = f"{r['w']}x{r['h']}" if (r.get('w') and r.get('h')) else "-"
        duration_str = human(r['duration_s'])
        print(f"{r['path']},{human(r['fps'])},{r['n_frames'] or '-'},\"{duration_str}\",{res},{r['status']}")

    if len(rows) > args.show_n:
        print(f"... ({len(rows)-args.show_n} mas filas no mostradas)")

    total = len(rows)
    fps_vals = [float(f"{r['fps']:.3f}") for r in rows if isinstance(r.get("fps"), (int, float))]
    dur_vals = [r["duration_s"] for r in rows if isinstance(r.get("duration_s"), (int, float))]

    print(f"\n[RESUMEN] Total: {total}")
    if fps_vals:
        print(f"[RESUMEN] FPS: min={min(fps_vals)} mediana={sorted(fps_vals)[len(fps_vals)//2]} max={max(fps_vals)}")
    if dur_vals:
        dur_vals_sorted = sorted(dur_vals)
        print(f"[RESUMEN] Duracion(s): min={dur_vals_sorted[0]:.3f} mediana={dur_vals_sorted[len(dur_vals_sorted)//2]:.3f} max={dur_vals_sorted[-1]:.3f}")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[INFO] CSV guardado en: {out}")

if __name__ == "__main__":
    main()