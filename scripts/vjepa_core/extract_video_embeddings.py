# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extractor de embeddings clip-a-clip para V-JEPA2 (u otros HF) desde:
  - Vídeos (CCD/DAD/flat)  o
  - Carpetas de FRAMES (DoTA, 10 fps nativo).

Modos dataset: auto | ccd | flat | frames | template
- frames/template: requiere --path_template (p.ej. "{root}/{id}/images")
- Negativos "__pre": recorta ROI <= (t_event_base - margin) para evitar fuga.

 Use --save tokens
Ejemplos de uso al final del archivo.
"""
from __future__ import annotations
import argparse, csv, glob, os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

# ------------------------- CSV utils -------------------------
def _float_or_none(x: Any):
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","none","null"}: return None
    try: return float(s.replace(",", "."))
    except: return None

def read_rows(csv_path: str) -> List[Dict[str, Any]]:
    out = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"id","target","time_of_event","time_of_alert"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"CSV debe tener columnas {sorted(need)}; encontrado {r.fieldnames}")
        for x in r:
            vid = (x.get("id") or "").strip()
            if not vid: continue
            out.append({
                "id": vid,
                "target": int(float(x.get("target","0") or 0)),
                "time_of_event": _float_or_none(x.get("time_of_event","")),
                "time_of_alert": _float_or_none(x.get("time_of_alert","")),
            })
    if not out: raise ValueError("CSV vacío.")
    return out

def build_event_time_map(rows: List[Dict[str,Any]]) -> Dict[str, float]:
    """Mapea id_base -> time_of_event (para negativos '__pre')."""
    m: Dict[str,float] = {}
    for r in rows:
        vid = r["id"]
        if r["target"] == 1 and r["time_of_event"] is not None:
            m[vid] = float(r["time_of_event"])
    return m

# ------------------------- Path resolvers -------------------------
VIDEO_EXTS_DEFAULT = (".mp4",".mkv",".avi",".mov",".MP4",".MOV")

def detect_zero_pad_ccd(videos_root: str, subdirs=("Crash-1500","Normal"), default=5, exts=VIDEO_EXTS_DEFAULT):
    pads=[]
    for sd in subdirs:
        d=os.path.join(videos_root,sd)
        if not os.path.isdir(d): continue
        for name in sorted(os.listdir(d)):
            low=name.lower()
            if any(low.endswith(e.lower()) for e in exts):
                stem=os.path.splitext(name)[0]
                if stem.isdigit(): pads.append(len(stem)); break
    return max(pads) if pads else default

def resolve_video_path_ccd(videos_root: str, uid: str, pad: int, exts=VIDEO_EXTS_DEFAULT):
    # CCD-like
    if "-" in uid:
        cls, stem = uid.split("-",1)
        sub = "Crash-1500" if cls.upper()=="C" else "Normal"
        for base in (stem, stem.zfill(pad)):
            for e in exts:
                p=os.path.join(videos_root, sub, f"{base}{e}")
                if os.path.exists(p): return p
    # Plano, ej: "positive/000001" o "000001"
    for e in exts:
        p = os.path.join(videos_root, f"{uid}{e}")
        if os.path.exists(p): return p
    # Si hay subcarpeta + zfill
    dirpart, name = os.path.split(uid)
    stem = os.path.splitext(name)[0]
    bases = [stem] + ([stem.zfill(pad)] if stem.isdigit() else [])
    for base in bases:
        for e in exts:
            p = os.path.join(videos_root, dirpart, f"{base}{e}")
            if os.path.exists(p): return p
    # Sin subcarpeta, número simple
    if uid.isdigit():
        for e in exts:
            p = os.path.join(videos_root, f"{uid.zfill(pad)}{e}")
            if os.path.exists(p): return p
    return None

def resolve_by_template(root: str, split: str, vid_id: str, template: str) -> str:
    base_id = vid_id.split("__pre")[0]
    return template.format(root=root, split=split, id=base_id)

# ------------------------- Video / Frames samplers -------------------------
try:
    import decord
    HAVE_DECORD=True
except Exception:
    HAVE_DECORD=False

class VideoSampler:
    def __init__(self, path: str, fps_target: float):
        self.path=path; self.fps_target=fps_target; self.backend=None
        self.native_fps=30.0; self.nframes=0; self.duration=0.0
        if HAVE_DECORD:
            try:
                decord.bridge.set_bridge('torch')
                self.vr=decord.VideoReader(path)
                self.native_fps=float(self.vr.get_avg_fps() or 30.0)
                self.nframes=len(self.vr); self.backend="decord"
            except Exception:
                self.vr=None
        if self.backend is None:
            from torchvision.io import read_video
            v,_,info = read_video(path, pts_unit="sec")
            if v.numel()==0: raise RuntimeError("No se pudo leer el vídeo.")
            self.v=v; self.native_fps=float(info.get("video_fps",30.0))
            self.nframes=int(v.shape[0]); self.backend="torchvision"
        self.duration=self.nframes/max(self.native_fps,1e-6)
        self.times=np.arange(0.0, self.duration, 1.0/self.fps_target, dtype=np.float32)
        self.idx=np.clip((self.times*self.native_fps).round().astype(np.int64), 0, max(self.nframes-1,0))

    def _get_batch_numpy(self, indices: np.ndarray) -> np.ndarray:
        if self.backend=="decord":
            batch=self.vr.get_batch(indices)
            try: return batch.asnumpy()
            except Exception: return batch.cpu().numpy()
        else:
            import torch as _t
            return self.v[_t.from_numpy(indices)].numpy()

# Frames sampler (carpeta con imágenes dentro)
try:
    import cv2; HAVE_CV2=True
except Exception:
    HAVE_CV2=False
from PIL import Image

class FramesSampler:
    def __init__(self, frames_dir: str, fps_target: float, native_fps: float = 10.0):
        # frames_dir debe ser el directorio que CONTIENE directamente las imágenes (*.jpg/*.png)
        self.frames_dir = frames_dir
        self.native_fps = float(native_fps)
        self.files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) +
                            glob.glob(os.path.join(frames_dir, "*.png")))
        if not self.files:
            # Si nos pasaron el dir del vídeo (sin 'images'), prueba a añadir '/images'
            alt = os.path.join(frames_dir, "images")
            self.files = sorted(glob.glob(os.path.join(alt, "*.jpg")) +
                                glob.glob(os.path.join(alt, "*.png")))
            if not self.files:
                raise RuntimeError(f"Sin imágenes en {frames_dir} (ni en {alt})")
            self.frames_dir = alt
        self.nframes = len(self.files)
        self.duration = self.nframes / max(self.native_fps, 1e-6)
        self.fps_target = float(fps_target)
        self.times = np.arange(0.0, self.duration, 1.0/self.fps_target, dtype=np.float32)
        self.idx = np.clip((self.times*self.native_fps).round().astype(np.int64), 0, self.nframes-1)

    def _get_batch_numpy(self, indices: np.ndarray) -> np.ndarray:
        frames = []
        for i in indices.tolist():
            f = self.files[int(i)]
            if HAVE_CV2:
                im = cv2.imread(f, cv2.IMREAD_COLOR)
                if im is None: raise RuntimeError(f"No puedo leer {f}")
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            else:
                im = np.array(Image.open(f).convert("RGB"))
            frames.append(im)
        return np.stack(frames, axis=0)

# ------------------------- Clip utils -------------------------
def compute_clip_indices(n_frames: int, clip_len: int, stride: int) -> List[Tuple[int,int]]:
    out=[]; t=0
    while t+clip_len <= n_frames:
        out.append((t, t+clip_len)); t+=stride
    return out

# ------------------------- HF model -------------------------
def load_hf(hf_repo: str, device: str, fp16: bool, bf16: bool=False):
    from transformers import AutoVideoProcessor, AutoModel
    import torch as _t
    if fp16 and bf16: raise ValueError("Usa fp16 O bf16, no ambos.")
    dtype = _t.float16 if (fp16 and device.startswith("cuda")) else (_t.bfloat16 if (bf16 and device.startswith("cuda")) else _t.float32)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    model = AutoModel.from_pretrained(hf_repo, dtype=dtype).to(device).eval()
    return processor, model, dtype

def get_vit_geometry_from_hf(model, img_size: int, frames_per_clip: int):
    """
    Intenta leer patch_size/tubelet_size desde model.config (HF).
    Si no existen, cae a defaults razonables (16/2).
    Devuelve dict con grid_h, grid_w, tubelet_size, frames_per_clip, patch_size y L esperado.
    """
    cfg = getattr(model, "config", None)
    patch = int(getattr(cfg, "patch_size", 16))
    tube = int(getattr(cfg, "tubelet_size", 2))
    grid = img_size // patch
    steps = int(frames_per_clip // tube)
    L_expected = steps * grid * grid
    return {
        "patch_size": patch,
        "tubelet_size": tube,
        "grid_h": grid,
        "grid_w": grid,
        "frames_per_clip": int(frames_per_clip),
        "L_expected": int(L_expected),
    }

@torch.no_grad()
def encode_clips(processor, model, clips_batch, img_size: int, device: str, run_dtype: torch.dtype, return_tokens: bool):
    """
    Hace una sola forward y devuelve:
      - pooled embedding (vector por clip)
      - y opcionalmente tokens (last_hidden_state) sin promediar.
    """
    inputs = processor(clips_batch, return_tensors="pt", size={"height": img_size, "width": img_size})
    inputs = {k:(v.to(device) if not torch.is_floating_point(v) else v.to(device=device, dtype=torch.float32))
              for k,v in inputs.items()}
    use_autocast = device.startswith("cuda") and run_dtype in (torch.float16, torch.bfloat16)
    ctx = torch.autocast("cuda", dtype=run_dtype) if use_autocast else torch.no_grad()
    with ctx:
        out = model(**inputs)

    # --- pooled (compat con tu flujo actual)
    if hasattr(out,"video_embeds") and out.video_embeds is not None: pooled=out.video_embeds
    elif hasattr(out,"pooler_output") and out.pooler_output is not None: pooled=out.pooler_output
    else:
        x_any = None
        if hasattr(out,"last_hidden_state") and out.last_hidden_state is not None:
            x_any = out.last_hidden_state
        elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
            x_any = out[0]
        elif hasattr(out,"logits") and isinstance(out.logits, torch.Tensor):
            x_any = out.logits
        if x_any is None:
            for v in out.__dict__.values():
                if isinstance(v, torch.Tensor): x_any=v; break
            if x_any is None: raise RuntimeError("No se pudo extraer embedding del modelo.")
        pooled = x_any.mean(1) if x_any.ndim==3 else x_any

    if not return_tokens:
        return pooled.float().cpu(), None

    # --- tokens sin promediar
    tokens = getattr(out, "last_hidden_state", None)
    if tokens is None or tokens.ndim != 3:
        raise RuntimeError("El modelo HF no expone last_hidden_state (tokens) o su forma no es (B,L,D).")
    return pooled.float().cpu(), tokens.float().cpu()

# (compat) tu función anterior; ya no la usamos cuando --save_tokens, pero la dejo por si la llamas en otro lado.
@torch.no_grad()
def embed_clips(processor, model, clips_batch, img_size: int, device: str, run_dtype: torch.dtype) -> torch.Tensor:
    inputs = processor(clips_batch, return_tensors="pt", size={"height": img_size, "width": img_size})
    inputs = {k:(v.to(device) if not torch.is_floating_point(v) else v.to(device=device, dtype=torch.float32))
              for k,v in inputs.items()}
    use_autocast = device.startswith("cuda") and run_dtype in (torch.float16, torch.bfloat16)
    ctx = torch.autocast("cuda", dtype=run_dtype) if use_autocast else torch.no_grad()
    with ctx:
        out = model(**inputs)

    if hasattr(out,"video_embeds") and out.video_embeds is not None: emb=out.video_embeds
    elif hasattr(out,"pooler_output") and out.pooler_output is not None: emb=out.pooler_output
    elif hasattr(out,"last_hidden_state") and out.last_hidden_state is not None:
        x=out.last_hidden_state; emb=x.mean(1) if x.ndim==3 else x
    elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
        x=out[0]; emb=x.mean(1) if x.ndim==3 else x
    elif hasattr(out,"logits") and isinstance(out.logits, torch.Tensor):
        x=out.logits; emb=x.mean(1) if x.ndim==3 else x
    else:
        x=None
        for v in out.__dict__.values():
            if isinstance(v, torch.Tensor): x=v; break
        if x is None: raise RuntimeError("No se pudo extraer embedding del modelo.")
        emb=x.mean(1) if x.ndim==3 else x

    return emb.float().cpu()

# ------------------------- CLI -------------------------
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--videos_root", default=".", help="Raíz de vídeos (CCD/DAD/flat)")
    ap.add_argument("--frames_root", default="", help="Raíz de frames (DoTA, etc.)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset", choices=["auto","ccd","flat","template","frames"], default="auto")
    ap.add_argument("--path_template", default="", help="p.ej. '{root}/{id}/images'  (frames/template)")
    ap.add_argument("--fps", type=float, default=10.0, help="FPS objetivo (submuestreo)")
    ap.add_argument("--native_fps", type=float, default=10.0, help="FPS nativo de frames (DoTA=10)")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--clip_stride", type=int, default=1)
    ap.add_argument("--batch_clips", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--pad_last", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--hist_s", type=float, default=5.0)
    ap.add_argument("--post_s", type=float, default=0.0)
    ap.add_argument("--anchor", choices=["alert","event","alert_then_event","event_then_alert"], default="event")
    ap.add_argument("--margin", type=float, default=0.3, help="Margen para negativos __pre respecto a t_event_base")
    ap.add_argument("--neg_strategy", choices=["full","head","tail","rand"], default="head")
    ap.add_argument("--neg_window_s", type=float, default=13.0)
    ap.add_argument("--neg_max_clips", type=int, default=0)  # 0 = todos
    ap.add_argument("--hf_repo", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save_tokens", action="store_true",
                    help="Guardar tokens (last_hidden_state) + metadatos ViT para attentive probe 1:1")
    return ap.parse_args()

# ------------------------- Main -------------------------
def main():
    args=parse_args()
    rng=np.random.default_rng(args.seed)
    rows=read_rows(args.csv)
    out_dir=Path(args.out_root)/args.split; out_dir.mkdir(parents=True, exist_ok=True)

    # Mapa id_base -> t_event (para negativos __pre)
    event_map = build_event_time_map(rows)

    # dataset detect
    dataset=args.dataset
    if dataset=="auto":
        is_ccd = os.path.isdir(os.path.join(args.videos_root,"Crash-1500")) and os.path.isdir(os.path.join(args.videos_root,"Normal"))
        if is_ccd:
            dataset="ccd"
        elif args.frames_root and os.path.isdir(args.frames_root):
            dataset="frames"
        else:
            dataset="flat"

    pad = detect_zero_pad_ccd(args.videos_root) if dataset=="ccd" else 5

    if args.dummy:
        processor=model=None; run_dtype=torch.float32; torch.manual_seed(0)
        print("[INFO] Dummy mode.")
    else:
        processor, model, run_dtype = load_hf(args.hf_repo, args.device, fp16=args.fp16, bf16=args.bf16)

    for r in rows:
        vid=r["id"]; target=int(r["target"]); t_event=r["time_of_event"]; t_alert=r["time_of_alert"]
        out_pt=out_dir/f"{vid}.pt"
        if args.skip_existing and out_pt.exists():
            print(f"[skip] {vid}"); continue
        out_pt.parent.mkdir(parents=True, exist_ok=True)

        # --- Elegir sampler según dataset ---
        vpath=None; frames_dir=None; sampler=None
        if dataset in ("frames","template"):
            if not args.path_template:
                # por defecto estructura plana con subcarpeta 'images'
                args.path_template = "{root}/{id}/images"
            frames_dir = resolve_by_template(args.frames_root or args.videos_root, args.split, vid, args.path_template)
            sampler=FramesSampler(frames_dir, fps_target=args.fps, native_fps=args.native_fps)
            backend="frames"
            native_fps=args.native_fps
        else:
            vpath = resolve_video_path_ccd(args.videos_root, vid, pad)
            if vpath is None:
                print(f"[WARN] no vídeo {vid}"); continue
            sampler=VideoSampler(vpath, fps_target=args.fps)
            backend=sampler.backend
            native_fps=sampler.native_fps

        duration=float(sampler.duration)

        # -------- ROI temporal y ancla --------
        if target==1:
            # Positivo: ancla según flag
            if args.anchor in ("event","event_then_alert"):
                anchor = t_event if t_event is not None else t_alert
            else:
                anchor = t_alert if t_alert is not None else t_event
            if anchor is None: anchor = duration/2.0
            start_s = max(0.0, float(anchor) - args.hist_s)
            end_s   = min(duration, float(anchor) + args.post_s)
            anchor_used = "event" if (anchor==t_event or args.anchor.startswith("event")) else "alert"
        else:
            # Negativo: __pre -> usar t_event del id base para NO cruzar el evento
            base_id = vid.split("__pre")[0]
            tev_base = event_map.get(base_id, None)
            if args.neg_strategy=="full":
                start_s, end_s = 0.0, duration
            elif args.neg_strategy=="tail":
                end_s=duration; start_s=max(0.0, end_s-args.neg_window_s)
            elif args.neg_strategy=="rand":
                win=args.neg_window_s
                if duration<=win: start_s,end_s=0.0,duration
                else:
                    s=float(rng.uniform(0.0, duration-win)); start_s,end_s=s,s+win
            else:  # head (por defecto)
                if tev_base is not None:
                    end_cap = max(0.0, tev_base - float(args.margin))
                    end_s = min(duration, min(args.neg_window_s, end_cap))
                    start_s = 0.0
                    if end_s < args.hist_s:  # no hay espacio real para una ventana completa
                        print(f"[neg-skip] {vid}: tev_base={tev_base:.3f}s < hist+margin => sin negativo util")
                        continue
                else:
                    # si no tenemos tev_base (no debería pasar con nuestros CSV), capea a neg_window_s
                    start_s, end_s = 0.0, min(duration, args.neg_window_s)
            anchor=float(end_s); anchor_used="neg_end"

        # -------- Submuestreo uniforme al fps deseado --------
        mask=(sampler.times>=start_s) & (sampler.times<end_s)
        times=sampler.times[mask]; idx=sampler.idx[mask]
        if len(idx)==0:
            print(f"[{args.split}] {vid}: ventana vacía [{start_s:.2f},{end_s:.2f})"); continue

        # -------- Ventanas de clips --------
        windows=compute_clip_indices(len(idx), args.clip_len, args.clip_stride)
        if target==0 and args.neg_max_clips>0 and len(windows)>args.neg_max_clips:
            sel=np.linspace(0,len(windows)-1,args.neg_max_clips,dtype=int).tolist()
            windows=[windows[i] for i in sel]
        if not windows:
            print(f"[{args.split}] {vid}: sin clips tras ventana"); continue

        print(f"[{args.split}] {vid}: {len(windows)} clips  L={args.clip_len} stride={args.clip_stride}  "
              f"ROI[{start_s:.2f},{end_s:.2f})  backend={backend}")

        # -------- Embedding por lotes --------
        def iter_clip_batches():
            for i0 in range(0,len(windows), args.batch_clips):
                this=windows[i0:i0+args.batch_clips]
                flat=[]; slices=[]; centers=[]
                for (s,e) in this:
                    clip_idx=idx[s:e]
                    if len(clip_idx)<args.clip_len:
                        if args.pad_last:
                            pad_val=clip_idx[-1]
                            pad=np.full((args.clip_len-len(clip_idx),), pad_val, dtype=clip_idx.dtype)
                            clip_idx=np.concatenate([clip_idx,pad],0)
                        else:
                            continue
                    a=len(flat); flat.extend(clip_idx.tolist()); b=len(flat)
                    slices.append((a,b))
                    centers.append(float(times[s:e].mean()))
                if not slices: continue
                frames_np = sampler._get_batch_numpy(np.array(flat, dtype=np.int64))
                batch=[]
                for (a,b) in slices:
                    arr=frames_np[a:b]   # (T,H,W,3)
                    batch.append([arr[j] for j in range(arr.shape[0])])
                yield batch, np.array(centers, dtype=np.float32)

        z_list=[]; tcent_list=[]; tok_list=[]
        if args.dummy:
            D=1024
            grid = args.img_size // 16
            tube = 2
            steps = int(args.clip_len // tube)
            Lsim = grid*grid*steps
            for clips_batch, tcenters in iter_clip_batches():
                z_list.append(np.random.standard_normal((len(clips_batch),D)).astype(np.float32))
                tcent_list.append(tcenters)
                if args.save_tokens:
                    tok_list.append(np.random.standard_normal((len(clips_batch), Lsim, D)).astype(np.float32))
            vit_geom = {
                "patch_size": 16, "tubelet_size": tube, "grid_h": grid, "grid_w": grid,
                "frames_per_clip": int(args.clip_len), "L_expected": int(Lsim),
            }
        else:
            vit_geom = get_vit_geometry_from_hf(model, args.img_size, args.clip_len)
            L_expected = vit_geom["L_expected"]
            for clips_batch, tcenters in iter_clip_batches():
                if not clips_batch: continue
                pooled, tokens = encode_clips(processor, model, clips_batch, args.img_size, args.device, run_dtype,
                                              return_tokens=args.save_tokens)
                z_list.append(pooled.numpy()); tcent_list.append(tcenters)
                if args.save_tokens:
                    tok_np = tokens.numpy()  # (B,L,D)
                    if tok_np.shape[1] != L_expected:
                        print(f"[WARN] tokens L={tok_np.shape[1]} != esperado {L_expected} "
                              f"(T={vit_geom['frames_per_clip']}, tubelet={vit_geom['tubelet_size']}, "
                              f"H/patch=W/patch={vit_geom['grid_h']}). "
                              f"Revisa img_size/patch_size o clip_len/tubelet_size.")
                    tok_list.append(tok_np)

        if not z_list:
            print(f"[{args.split}] {vid}: vacío tras embedding"); continue

        Zc=torch.from_numpy(np.concatenate(z_list,0))  # (N_clip,D)
        Tc=torch.from_numpy(np.concatenate(tcent_list,0).astype(np.float32))  # centros
        delta = Tc - float(anchor)

        payload={
            "id": vid,
            "z_clip": Zc,                   # (N,D)  baseline/compat
            "clip_times": Tc,               # centros en seg
            "delta_to_anchor": delta,       # Tc - anchor
            "fps_target": float(args.fps),
            "meta": {
                "target": target,
                "time_of_event": t_event,
                "time_of_alert": t_alert,
                "video_path": vpath,
                "frames_dir": frames_dir,
                "backend": backend,
                "duration": float(sampler.duration),
                "clip_len": int(args.clip_len),
                "clip_stride": int(args.clip_stride),
                "img_size": int(args.img_size),
                "hf_repo": str(args.hf_repo),
                "window": [float(start_s), float(end_s)],
                "anchor_used": anchor_used,
                "dataset": dataset,
                "native_fps": float(native_fps),
            }
        }

        if args.save_tokens:
            Tok = torch.from_numpy(np.concatenate(tok_list,0)).to(torch.float16)  # (N_clip,L,D) en fp16
            payload["tokens"] = Tok
            payload["tokens_meta"] = {
                "patch_size": vit_geom["patch_size"],
                "tubelet_size": vit_geom["tubelet_size"],
                "grid_h": vit_geom["grid_h"],
                "grid_w": vit_geom["grid_w"],
                "frames_per_clip": vit_geom["frames_per_clip"],
                "L_expected": vit_geom["L_expected"],
                "dtype": "float16",
            }

        torch.save(payload, out_pt)
        print(f"[OK] {vid} -> {out_pt}")

if __name__=="__main__":
    main()
