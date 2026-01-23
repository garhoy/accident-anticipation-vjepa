#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCD meta + splits con IDs únicos por clase (C-/N-).
- Lee Crash-1500.txt para positivos (bins @10fps -> event_time_s).
- Lee train.txt/test.txt oficiales con prefijos positive/negative.
- Construye unique_id = 'C-000123' o 'N-000987' para evitar falsos duplicados.
- Genera val estratificado a partir de train (semilla fija).
- Marca la columna 'split' en meta.csv (train/val/test).
"""

import argparse, csv, re, sys, random
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import yaml

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------- Utils ----------------

def parse_bins(bin_str: str) -> List[int]:
    nums = re.findall(r"-?\d+", bin_str)
    if len(nums) != 50:
        raise ValueError(f"Se esperaban 50 bins, recibido len={len(nums)}")
    return [int(x) for x in nums]

def first_one_index(bins: List[int]) -> Optional[int]:
    for i, v in enumerate(bins):
        if v == 1:
            return i
    return None

def list_videos(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out[p.stem] = p
    return out

def make_uid(cls_code: str, stem: str) -> str:
    # cls_code: 'C' (Crash) o 'N' (Normal)
    return f"{cls_code}-{stem}"

def uid_parts(uid: str) -> Tuple[str, str]:
    # 'C-000123' -> ('C','000123')
    cls, stem = uid.split("-", 1)
    return cls, stem

def read_official_uids(txt: Path, pos_map: Dict[str, Path], neg_map: Dict[str, Path]) -> List[str]:
    """
    Acepta líneas tipo:
        'negative/001355.npz 0'
        'positive/000713.npz 1'
        'positive/000713.npz'
        '001234.npz'  (sin prefijo; intentamos inferir por presencia en carpetas)
    Devuelve uids como 'N-001355', 'C-000713', ...
    """
    uids: List[str] = []
    for raw in txt.read_text().strip().splitlines():
        s = raw.strip()
        if not s:
            continue
        token = s.split()[0]        # primer token ignora etiqueta final
        p = Path(token)
        stem = p.stem               # '001355'
        low = token.lower()

        if "positive" in low:
            uid = make_uid("C", stem)
        elif "negative" in low:
            uid = make_uid("N", stem)
        else:
            # Intento de inferencia si no hay prefijo
            in_pos = stem in pos_map
            in_neg = stem in neg_map
            if in_pos and not in_neg:
                uid = make_uid("C", stem)
            elif in_neg and not in_pos:
                uid = make_uid("N", stem)
            elif in_pos and in_neg:
                # Ambiguo: por seguridad preferimos positivos si existe en Crash
                uid = make_uid("C", stem)
                sys.stderr.write(f"[WARN] Línea ambigua sin prefijo para id={stem}; "
                                 f"infiere 'C-'. Considera arreglar la lista: {s}\n")
            else:
                sys.stderr.write(f"[WARN] id={stem} no existe en pos/neg. Se omite.\n")
                continue
        uids.append(uid)
    return uids

def write_list(lines: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in lines:
            f.write(f"{x}\n")

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta a configs/ccd.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds   = cfg["dataset"]
    root = Path(ds["root"])
    pos_dir = Path(ds["positives_dir"])
    neg_dir = Path(ds["negatives_dir"])
    crash_txt = Path(ds["crash_txt"])
    meta_csv  = Path(ds["meta_csv"])

    use_official = bool(ds.get("use_official_splits", False))
    off_train = Path(ds.get("official_train_list", "")) if use_official else None
    off_test  = Path(ds.get("official_test_list", ""))  if use_official else None

    splits = ds["splits"]
    train_ids_out = Path(splits["train_ids"])
    val_ids_out   = Path(splits["val_ids"])
    test_ids_out  = Path(splits["test_ids"])
    val_size = float(ds.get("make_val_size", 0.2))
    seed     = int(ds.get("seed", 42))

    fps_embed = float(cfg.get("extract", {}).get("fps", 10))
    dataset_version = ds.get("dataset_version", "official")

    # Checks
    assert crash_txt.exists(), f"No existe: {crash_txt}"
    assert pos_dir.exists(),   f"No existe: {pos_dir}"
    assert neg_dir.exists(),   f"No existe: {neg_dir}"
    meta_csv.parent.mkdir(parents=True, exist_ok=True)

    # Indexar vídeos por clase
    pos_map = list_videos(pos_dir)  # { '000123': Path(...Crash-1500/000123.mp4) }
    neg_map = list_videos(neg_dir)  # { '000987': Path(...Normal/000987.mp4) }

    # ----- Parse Crash-1500.txt (positivos) -----
    pos_info: Dict[str, Dict] = {}  # id -> info (bins, event_frame, event_time_s)
    for line in crash_txt.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        first_comma = line.find(",")
        vid = line[:first_comma].strip()  # '000123'
        lbr = line.find("[", first_comma)
        rbr = line.find("]", lbr)
        if lbr < 0 or rbr < 0:
            raise ValueError(f"Fila malformada para id={vid}: {line}")
        bins_str = line[lbr:rbr+1]
        parts = [x.strip() for x in line[rbr+2:].split(",")]
        if len(parts) < 5:
            raise ValueError(f"Fila malformada para id={vid}: {line}")
        start_frame, youtube_id, timing, weather, ego = parts[:5]
        bins = parse_bins(bins_str)
        i_on = first_one_index(bins)
        event_time_s = "" if i_on is None else i_on / 10.0  # CCD bins @10fps

        pos_info[vid] = {
            "bins": bins,
            "event_frame": i_on if i_on is not None else "",
            "event_time_s": event_time_s,
            "timing": timing,
            "weather": weather,
            "ego_involve": ego,
        }

    # ----- Construir filas meta (sin split de momento) -----
    rows: List[Dict] = []
    uid_to_idx: Dict[str, int] = {}

    # Positivos (Crash-1500)
    for vid, p in sorted(pos_map.items()):
        rel_path = str(p.relative_to(root).as_posix())
        info = pos_info.get(vid, None)
        if info is None:
            sys.stderr.write(f"[WARN] Positivo {vid} no tiene entrada en Crash-1500.txt\n")
            bins = ["0"]*50
            event_frame = ""
            event_time_s = ""
            timing = weather = ego = ""
        else:
            bins = list(map(str, info["bins"]))
            event_frame = info["event_frame"]
            event_time_s = info["event_time_s"]
            timing = info["timing"]; weather = info["weather"]; ego = info["ego_involve"]

        uid = make_uid("C", vid)
        row = {
            "unique_id": uid,
            "id": vid,
            "rel_path": rel_path,
            "source": "CCD",
            "dataset_version": dataset_version,
            "split": "",  # se rellena luego
            "fps_src": 10,
            "n_frames_src": 50,
            "dur_src_s": 5.0,
            "target": 1,
            "event_frame": event_frame,
            "event_time_s": event_time_s,
            "fps_embed": fps_embed,
            "timing": timing,
            "weather": weather,
            "ego_involve": ego,
            "bins_10fps_50": " ".join(bins),
        }
        uid_to_idx[uid] = len(rows)
        rows.append(row)

    # Negativos (Normal)
    zero_bins = " ".join(["0"]*50)
    for vid, p in sorted(neg_map.items()):
        rel_path = str(p.relative_to(root).as_posix())
        uid = make_uid("N", vid)
        row = {
            "unique_id": uid,
            "id": vid,
            "rel_path": rel_path,
            "source": "CCD",
            "dataset_version": dataset_version,
            "split": "",  # se rellena luego
            "fps_src": "",         # desconocido en CCD-Original (trimmed externos)
            "n_frames_src": "",    # dejamos vacío
            "dur_src_s": "",       # dejamos vacío
            "target": 0,
            "event_frame": "",
            "event_time_s": "",
            "fps_embed": fps_embed,
            "timing": "",
            "weather": "",
            "ego_involve": "",
            "bins_10fps_50": zero_bins,
        }
        uid_to_idx[uid] = len(rows)
        rows.append(row)

    # ----- Sumarizar totales base -----
    n_pos = sum(1 for r in rows if r["target"] == 1)
    n_neg = sum(1 for r in rows if r["target"] == 0)
    print(f"[INFO] Videos indexados -> Pos:{n_pos}  Neg:{n_neg}  Total:{len(rows)}")

    # ----- Splits -----
    train_uids: List[str] = []
    test_uids: List[str]  = []

    if use_official and off_train and off_test and off_train.exists() and off_test.exists():
        off_train_uids = read_official_uids(off_train, pos_map, neg_map)
        off_test_uids  = read_official_uids(off_test,  pos_map, neg_map)

        # Filtrar a los que existen físicamente
        def keep_existing(uids: List[str]) -> List[str]:
            kept = []
            for uid in uids:
                if uid in uid_to_idx:
                    kept.append(uid)
                else:
                    cls, stem = uid_parts(uid)
                    sys.stderr.write(f"[WARN] {uid} no existe físicamente. Se omite.\n")
            return kept

        train_uids = keep_existing(off_train_uids)
        test_uids  = keep_existing(off_test_uids)

        overlap = set(train_uids) & set(test_uids)
        print(f"[INFO] Official splits -> train:{len(train_uids)}  test:{len(test_uids)}  overlap_uids:{len(overlap)}")
    else:
        # Fallback: si no hay listas oficiales, no hacemos split aquí.
        print("[INFO] No hay listas oficiales disponibles; omite generación de splits aquí.")
        train_uids, test_uids = [], []

    # ----- Hacer VAL estratificado desde TRAIN -----
    random.seed(seed)
    if train_uids:
        train_pos = [u for u in train_uids if u.startswith("C-")]
        train_neg = [u for u in train_uids if u.startswith("N-")]
        k_pos = int(round(len(train_pos) * val_size))
        k_neg = int(round(len(train_neg) * val_size))
        val_from_pos = set(random.sample(train_pos, k_pos)) if k_pos > 0 else set()
        val_from_neg = set(random.sample(train_neg, k_neg)) if k_neg > 0 else set()
        val_uids = val_from_pos | val_from_neg
        new_train = [u for u in train_uids if u not in val_uids]

        # Guardar listas (por UID único)
        write_list(sorted(new_train), train_ids_out)
        write_list(sorted(val_uids),   val_ids_out)
        write_list(sorted(test_uids),  test_ids_out)

        print(f"[INFO] Splits escritos -> train:{len(new_train)}  val:{len(val_uids)}  test:{len(test_uids)}")
        # Marcar 'split' en meta
        train_set = set(new_train)
        val_set   = set(val_uids)
        test_set  = set(test_uids)
        for uid, idx in uid_to_idx.items():
            if uid in train_set:
                rows[idx]["split"] = "train"
            elif uid in val_set:
                rows[idx]["split"] = "val"
            elif uid in test_set:
                rows[idx]["split"] = "test"
            else:
                rows[idx]["split"] = ""
    else:
        print("[WARN] No se generaron splits (faltan listas oficiales). 'split' queda vacío.")

    # ----- Escribir meta.csv -----
    fieldnames = [
        "unique_id","id","rel_path","source","dataset_version","split",
        "fps_src","n_frames_src","dur_src_s",
        "target","event_frame","event_time_s",
        "fps_embed",
        "timing","weather","ego_involve",
        "bins_10fps_50",
    ]
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] meta.csv -> {meta_csv} (filas={len(rows)})")

if __name__ == "__main__":
    main()
