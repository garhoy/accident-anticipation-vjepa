#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, argparse
from pathlib import Path

def load_meta(meta_csv: Path):
    rows=[]
    with meta_csv.open("r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for x in r:
            uid = (x.get("unique_id") or "").strip()   # 'C-000123' | 'N-000456'
            if not uid: 
                rel = x.get("rel_path","")
                stem = Path(rel).stem
                cls = "C" if "Crash" in rel or "Crash-1500" in rel else "N"
                uid = f"{cls}-{stem}"
            target = int(x.get("target","0") or 0)
            et = x.get("event_time_s","")
            time_of_event = float(et) if str(et).strip()!="" else None
            rows.append({
                "id": uid,
                "time_of_event": time_of_event,
                "time_of_alert": None,
                "target": target,
            })
    return rows

def load_uid_list(path: Path):
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}

def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["id","time_of_event","time_of_alert","target"])
        w.writeheader()
        for r in rows: w.writerow(r)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--val_ids", required=True)
    ap.add_argument("--test_ids", required=True)
    ap.add_argument("--out_dir", required=True, help="p.ej. data/metadata/CCD/extractor_csv")
    args=ap.parse_args()

    meta_rows = load_meta(Path(args.meta_csv))
    by_id = {r["id"]: r for r in meta_rows}

    for split, ids_path in [("train", args.train_ids), ("val", args.val_ids), ("test", args.test_ids)]:
        ids = load_uid_list(Path(ids_path))
        out = [by_id[u] for u in ids if u in by_id]
        out_csv = Path(args.out_dir) / f"{split}.csv"
        write_csv(out, out_csv)
        print(f"[OK] {split}: {len(out)} filas -> {out_csv}")

if __name__ == "__main__":
    main()
