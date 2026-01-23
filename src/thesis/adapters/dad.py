#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAD adapter (como CCD), versión 5s:
- Crea meta.csv con IDs únicos y tiempos de evento al FINAL DEL CLIP (≈duración).
- Hace VAL estratificado desde TRAIN (seed fija).
- Genera CSVs de extracción separados: training_train.csv, training_val.csv, testing.csv.
- Imprime DEBUG de duración y fps por split/clase.

Requiere: ffprobe, PyYAML.
"""
from __future__ import annotations
import argparse, csv, random, subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

VIDEO_EXTS = {".mp4",".avi",".mov",".mkv",".webm",".MP4",".MOV"}

def ffprobe_num(path:str, entries:str)->float:
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0","-show_entries",entries,"-of","default=nw=1:nk=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        s = r.stdout.strip()
        if "/" in s:
            a,b = s.split("/",1)
            return float(a) / max(float(b), 1.0)
        return float(s)
    except:
        return 0.0

def video_info(p:Path)->Tuple[float,float,int]:
    dur = ffprobe_num(str(p), "format=duration")
    fps = ffprobe_num(str(p), "stream=avg_frame_rate")
    nb  = ffprobe_num(str(p), "stream=nb_frames")
    if nb > 0:
        nframes = int(round(nb))
    elif (dur > 0 and fps > 0):
        nframes = int(round(dur * fps))
    else:
        nframes = 0
    return dur, fps, nframes

def list_videos(folder:Path)->Dict[str,Path]:
    out={}
    # prioridad .mp4
    for p in folder.glob("*.mp4"):
        out[p.stem] = p
    # resto de extensiones
    for p in folder.iterdir():
        if p.is_file() and p.suffix in VIDEO_EXTS and p.suffix != ".mp4":
            out[p.stem] = p
    return out

def make_uid(split_code:str, cls:str, stem:str)->str:
    # split_code in {"TR","TE"}, cls in {"C","N"}
    return f"{cls}-{split_code}-{stem}"

def write_lines(lines:List[str], path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in lines: f.write(f"{x}\n")

def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except:
        return ""

def _print_stats(rows: List[Dict]):
    from statistics import mean
    def stats(vals):
        if not vals: return None
        return (min(vals), mean(vals), max(vals))
    for sp in ["train","val","test"]:
        for lab, name in [(1,"pos"),(0,"neg")]:
            rr = [r for r in rows if r["split"]==sp and r["target"]==lab]
            if not rr:
                continue
            durs = [float(r["dur_src_s"]) for r in rr if r["dur_src_s"]]
            fpss = [float(r["fps_src"])   for r in rr if r["fps_src"]]
            sd = stats(durs); sf = stats(fpss)
            print(f"[DEBUG] {sp}/{name}: n={len(rr)} "
                  f"dur_s[min/mean/max]={None if not sd else tuple(round(x,3) for x in sd)} "
                  f"fps[min/mean/max]={None if not sf else tuple(round(x,3) for x in sf)}")
            bad_dur = [d for d in durs if abs(d-5.0) > 0.15]  # tolerancia 150ms
            bad_fps = [f for f in fpss if abs(f-20.0) > 0.6 and abs(f-10.0) > 0.6]
            if bad_dur:
                print(f"  -> WARN: {len(bad_dur)} vídeos lejos de 5.0s (±0.15s).")
            if bad_fps:
                print(f"  -> WARN: {len(bad_fps)} vídeos con fps no esperados (ni ~10 ni ~20).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg  = yaml.safe_load(Path(args.config).read_text())
    ds   = cfg["dataset"]
    root = Path(ds["root"])  # ej: data/raw/DAD/videos (convertidos a 5s)
    pos_name = ds.get("pos_name","positive")
    neg_name = ds.get("neg_name","negative")

    meta_csv = Path(ds["meta_csv"])
    splits   = ds["splits"]
    train_ids_out = Path(splits["train_ids"])
    val_ids_out   = Path(splits["val_ids"])
    test_ids_out  = Path(splits["test_ids"])
    val_size = float(ds.get("make_val_size",0.2))
    seed     = int(ds.get("seed",42))

    extractor_csv = ds["extractor_csv"]
    tr_train_csv = Path(extractor_csv["training_train"])
    tr_val_csv   = Path(extractor_csv["training_val"])
    te_csv       = Path(extractor_csv["testing"])

    # carpetas esperadas
    tr_pos = root/"training"/pos_name
    tr_neg = root/"training"/neg_name
    te_pos = root/"testing"/pos_name
    te_neg = root/"testing"/neg_name
    for p in [tr_pos,tr_neg,te_pos,te_neg]:
        assert p.exists(), f"Falta carpeta: {p}"

    # index
    trp = list_videos(tr_pos); trn = list_videos(tr_neg)
    tep = list_videos(te_pos); ten = list_videos(te_neg)

    rows: List[Dict] = []
    uid2idx: Dict[str,int] = {}

    def add_rows(split_code:str, mapping:Dict[str,Path], target:int):
        for stem, p in sorted(mapping.items()):
            cls = "C" if target==1 else "N"
            uid = make_uid(split_code, cls, stem)
            rel_path = p.relative_to(root).as_posix()  # p.ej training/positive/000123.mp4
            dur, fps, nf = video_info(p)

            # evento al FINAL del clip (≈ duración)
            t_event = dur if target==1 else ""

            row = {
                "unique_id": uid,
                "id": stem,
                "rel_path": rel_path,
                "source": "DAD",
                "dataset_version": "20fps5s",
                "split": "train" if split_code=="TR" else "test",  # temporal (ajustaremos VAL luego)
                "fps_src": _f(fps),
                "n_frames_src": str(nf) if nf>0 else "",
                "dur_src_s": _f(dur),
                "target": target,
                "event_frame": "",
                "event_time_s": _f(t_event) if t_event!="" else "",
                "fps_embed": str(cfg.get("extract",{}).get("fps",20)),
                "timing": "",
                "weather": "",
                "ego_involve": "",
                "bins_10fps_50": "",
            }
            uid2idx[uid] = len(rows)
            rows.append(row)

    add_rows("TR", trp, 1); add_rows("TR", trn, 0)
    add_rows("TE", tep, 1); add_rows("TE", ten, 0)

    # === Hacer VAL estratificado desde TRAIN ===
    random.seed(seed)
    train_uids = [r["unique_id"] for r in rows if r["split"]=="train"]
    pos_train  = [u for u in train_uids if u.startswith("C-")]
    neg_train  = [u for u in train_uids if u.startswith("N-")]
    kpos = int(round(len(pos_train)*val_size))
    kneg = int(round(len(neg_train)*val_size))
    val_set = set(random.sample(pos_train, kpos) + random.sample(neg_train, kneg))
    new_train = [u for u in train_uids if u not in val_set]
    test_uids = [r["unique_id"] for r in rows if r["split"]=="test"]

    # Guardar TXT reproducibles
    write_lines(sorted(new_train), train_ids_out)
    write_lines(sorted(val_set),   val_ids_out)
    write_lines(sorted(test_uids), test_ids_out)

    # Actualizar split real
    trainS=set(new_train); valS=set(val_set); testS=set(test_uids)
    for uid, idx in uid2idx.items():
        if uid in trainS: rows[idx]["split"]="train"
        elif uid in valS: rows[idx]["split"]="val"
        elif uid in testS: rows[idx]["split"]="test"

    # === meta.csv
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "unique_id","id","rel_path","source","dataset_version","split",
        "fps_src","n_frames_src","dur_src_s",
        "target","event_frame","event_time_s",
        "fps_embed","timing","weather","ego_involve","bins_10fps_50"
    ]
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    n_train = sum(1 for r in rows if r["split"]=="train")
    n_val   = sum(1 for r in rows if r["split"]=="val")
    n_test  = sum(1 for r in rows if r["split"]=="test")
    print(f"[OK] meta.csv -> {meta_csv} (rows={len(rows)})")
    print(f"[OK] splits   -> train:{n_train}  val:{n_val}  test:{n_test}")

    # === CSVs de extracción por split real (train/val/test) ===
    def _uid_to_idfield(uid:str)->str:
        cls = pos_name if uid.startswith("C-") else neg_name
        stem = uid.split("-")[-1]
        return f"{cls}/{stem}"

    def write_subset_csv(uids_set:set, out_csv:Path):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out=[]
        sel = set(uids_set)
        for r in rows:
            if r["unique_id"] not in sel:
                continue
            out.append({
                "id": _uid_to_idfield(r["unique_id"]),
                "time_of_event": r["event_time_s"] if r["target"]==1 else "",
                "time_of_alert": "",
                "target": r["target"],
            })
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id","time_of_event","time_of_alert","target"])
            w.writeheader(); w.writerows(out)
        n_pos = sum(1 for x in out if str(x["target"])=="1")
        n_neg = len(out) - n_pos
        print(f"[OK] extractor CSV -> {out_csv} (rows={len(out)} | pos={n_pos} neg={n_neg})")

    # train / val desde partición estratificada + test desde split test
    write_subset_csv(trainS, tr_train_csv)
    write_subset_csv(valS,   tr_val_csv)
    write_subset_csv(testS,  te_csv)

    # DEBUG extra
    _print_stats(rows)
    print("[DONE]")

if __name__=="__main__":
    main()
