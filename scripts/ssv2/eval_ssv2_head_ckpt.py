#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ssv2_head_ckpt.py

Evalúa Something-Something V2 usando un checkpoint entrenado con train_ssv2_head_badas.py

- Modelo base: AutoModelForVideoClassification (facebook/vjepa2-vitl-fpc16-256-ssv2)
- Se carga el state_dict del checkpoint (encoder + cabeza entrenada).
- No se toca el encoder_override aquí: se asume que el ckpt ya lleva dentro
  el encoder correcto (V-JEPA o BADAS) + la head entrenada.

Soporta:
  --split train / val / test
  Para 'test' usa test.json + test-answers.csv para obtener las etiquetas.

Uso típico:

  # Modelo A (encoder V-JEPA + head entrenada)
  python examples/eval_ssv2_head_ckpt.py \
    --hf_repo facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --ckpt weights_ssv2_vjepa_head/badas_ssv2_head_best.pt \
    --split val \
    --batch-size 64 \
    --num-workers 8

  # Modelo B (encoder BADAS + head entrenada)
  python examples/eval_ssv2_head_ckpt.py \
    --hf_repo facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --ckpt weights_ssv2_badas_head/badas_ssv2_head_best.pt \
    --split test \
    --batch-size 64 \
    --num-workers 8
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ----------------------------------------------------------------------
#  Utils: localizar V-JEPA 2 y resolver rutas
# ----------------------------------------------------------------------

def find_vjepa_root(start_path: Path, max_levels: int = 5) -> Optional[Path]:
    """
    Sube directorios hasta encontrar 'V-JEPA 2'.
    """
    current = start_path
    for _ in range(max_levels):
        candidate = current.parent / "V-JEPA 2"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    return None


def resolve_video_path(root: Path, vid: str) -> Optional[Path]:
    """
    SSv2 original: f'{id}.webm' en 20bn-something-something-v2.
    Dejamos fallback a otros formatos por si has recodificado.
    """
    exts = [".webm", ".mp4", ".avi", ".mkv", ".MOV", ".mov"]
    for ext in exts:
        p = root / f"{vid}{ext}"
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------------
#  Dataset SSv2 (usando TEMPLATE → CLASS_ID)
# ----------------------------------------------------------------------

class SomethingSomethingV2Dataset(Dataset):
    """
    Para train/val: usa annotation_file (train.json / validation.json)
      - Cada entrada tiene: id, label, template, placeholders
      - Usamos SOLO 'template' para mapear a la clase

    Para test:
      - test.json: lista de {"id": ...}
      - test-answers.csv: "id;template"
      - Volvemos a usar 'template' para mapear.
    """

    def __init__(
        self,
        split: str,
        annotation_file: Path,
        video_root: Path,
        template_to_id: Dict[str, int],
        num_frames: int,
        test_answers_csv: Optional[Path] = None,
        max_videos: int = -1,
    ):
        super().__init__()

        self.video_root = video_root
        self.template_to_id = template_to_id
        self.num_frames = num_frames

        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations no encontradas: {annotation_file}")
        if not video_root.exists():
            raise FileNotFoundError(f"Carpeta de vídeos no encontrada: {video_root}")

        with annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples: List[Tuple[str, int]] = []

        if split in {"train", "val"}:
            # JSON con campos: id, label, template, placeholders
            for row in data:
                vid = str(row.get("id", "")).strip()
                if not vid:
                    continue

                template = str(row.get("template", "")).strip()
                if not template:
                    continue

                if template not in self.template_to_id:
                    # Plantilla no conocida por el modelo HF
                    continue

                vpath = resolve_video_path(video_root, vid)
                if vpath is None:
                    # Vídeo no descargado / corrupto
                    continue

                cls_id = int(self.template_to_id[template])
                samples.append((str(vpath), cls_id))

                if max_videos > 0 and len(samples) >= max_videos:
                    break

        elif split == "test":
            # test.json + test-answers.csv (id;template)
            if test_answers_csv is None or not test_answers_csv.exists():
                raise FileNotFoundError(
                    f"Se necesita test-answers.csv para el split 'test': {test_answers_csv}"
                )

            # Construimos mapping id -> template desde el CSV
            id2template: Dict[str, str] = {}
            with test_answers_csv.open("r", encoding="utf-8") as fcsv:
                for line in fcsv:
                    line = line.strip()
                    if not line:
                        continue
                    # formato: id;Template text...
                    parts = line.split(";", 1)
                    if len(parts) != 2:
                        continue
                    vid_id, tmpl = parts[0].strip(), parts[1].strip()
                    id2template[vid_id] = tmpl

            for row in data:
                vid = str(row.get("id", "")).strip()
                if not vid:
                    continue

                tmpl = id2template.get(vid, "").strip()
                if not tmpl:
                    continue

                if tmpl not in self.template_to_id:
                    continue

                vpath = resolve_video_path(video_root, vid)
                if vpath is None:
                    continue

                cls_id = int(self.template_to_id[tmpl])
                samples.append((str(vpath), cls_id))

                if max_videos > 0 and len(samples) >= max_videos:
                    break
        else:
            raise ValueError(f"split desconocido: {split}")

        if len(samples) == 0:
            raise RuntimeError(
                f"Dataset vacío tras filtrar. Revisa JSONs, vídeos y mapping template→id."
            )

        self.samples = samples
        print(f"[DATA] SSv2 ({split}): {len(self.samples)} vídeos listos ({annotation_file.name})")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video_array(self, path: str):
        """
        Carga vídeo con torchvision.io.read_video.
        Devuelve np.array (T, H, W, C) uint8, recortado/padding a num_frames.
        """
        vframes, _, _ = read_video(path, pts_unit="sec")  # (T, H, W, C), uint8
        if vframes.numel() == 0:
            raise RuntimeError(f"Vídeo vacío: {path}")

        T = vframes.shape[0]
        if T >= self.num_frames:
            idx = torch.linspace(0, T - 1, self.num_frames).long()
            vframes = vframes[idx]
        else:
            pad = self.num_frames - T
            last = vframes[-1:].expand(pad, -1, -1, -1)
            vframes = torch.cat([vframes, last], dim=0)

        # Tensor -> numpy (T, H, W, C)
        return vframes.numpy()

    def __getitem__(self, idx: int):
        vpath, cls_id = self.samples[idx]
        video_np = self._load_video_array(vpath)
        return video_np, cls_id


# ----------------------------------------------------------------------
#  Collate usando AutoVideoProcessor
# ----------------------------------------------------------------------

def make_collate_fn(processor: AutoVideoProcessor):
    def collate_fn(batch: List[Tuple[Any, int]]):
        """
        batch: lista de (video_np[T,H,W,C], label)
        """
        videos_np, labels = zip(*batch)
        inputs = processor(list(videos_np), return_tensors="pt")
        labels_t = torch.tensor(labels, dtype=torch.long)
        return inputs, labels_t
    return collate_fn


# ----------------------------------------------------------------------
#  Evaluación
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, int]:
    """
    Devuelve: (top1, top5, num_samples)
    """
    model.eval()
    top1_correct = 0.0
    top5_correct = 0.0
    n_total = 0

    it = dataloader
    if tqdm is not None:
        it = tqdm(dataloader, desc="Eval SSv2 (ckpt)", unit="batch")

    for batch in it:
        inputs, labels = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        outputs = model(**inputs)
        logits = outputs.logits  # (B, num_classes)

        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
        pred = pred.t()  # (maxk, B)
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # (maxk, B)

        top1_correct += correct[:1].reshape(-1).float().sum().item()
        top5_correct += correct[:maxk].reshape(-1).float().sum().item()
        n_total += labels.size(0)

    top1 = 100.0 * top1_correct / max(1, n_total)
    top5 = 100.0 * top5_correct / max(1, n_total)
    return top1, top5, n_total


# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf_repo",
        type=str,
        default="facebook/vjepa2-vitl-fpc16-256-ssv2",
        help="Repo HF del modelo SSv2 oficial de V-JEPA 2",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Ruta al checkpoint entrenado (badas_ssv2_head_best.pt)",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split SSv2 a evaluar (train/val/test).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size de evaluación.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Num workers del DataLoader.",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=-1,
        help="Para debug: máximo nº de vídeos a evaluar (<=0 = todos).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' o 'cpu'. Por defecto usa CUDA si está disponible.",
    )
    args = ap.parse_args()

    # ----------------------------------------------------------
    # 0) Localizar raíz V-JEPA 2
    # ----------------------------------------------------------
    script_dir = Path(__file__).parent.resolve()
    vjepa_root = find_vjepa_root(script_dir)
    if vjepa_root is None:
        # Fallback manual
        vjepa_root = Path.home() / "Desktop/Master Thesis/Experiments/V-JEPA 2"
        if not vjepa_root.exists():
            print("[ERR] No se encuentra la carpeta 'V-JEPA 2'. Edita el script.")
            sys.exit(1)

    print(f"[OK] V-JEPA 2 root: {vjepa_root}")

    ssv2_raw_root = vjepa_root / "data/raw/SSv2"
    ssv2_meta_root = vjepa_root / "data/metadata/SSv2/labels"

    video_root = ssv2_raw_root / "20bn-something-something-v2"
    if not video_root.exists():
        alt = ssv2_raw_root / "Videos"
        if alt.exists():
            video_root = alt
        else:
            print(f"[ERR] No encuentro carpeta de vídeos SSv2 en {video_root} ni en {alt}")
            sys.exit(1)

    if args.split == "train":
        ann_file = ssv2_meta_root / "train.json"
    elif args.split == "val":
        ann_file = ssv2_meta_root / "validation.json"
    else:
        ann_file = ssv2_meta_root / "test.json"

    test_answers_csv = ssv2_meta_root / "test-answers.csv"

    print(f"[DATA] JSON anotaciones : {ann_file}")
    print(f"[DATA] Vídeos raíz      : {video_root}")

    # ----------------------------------------------------------
    # 1) Device
    # ----------------------------------------------------------
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[INFO] Device: {device}")

    # ----------------------------------------------------------
    # 2) Cargar modelo base HF
    # ----------------------------------------------------------
    print(f"[LOAD] Modelo HF base: {args.hf_repo}")
    model = AutoModelForVideoClassification.from_pretrained(args.hf_repo)
    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)

    num_frames = getattr(model.config, "num_frames", None)
    if num_frames is None:
        num_frames = getattr(model.config, "frames_per_clip", 16)
    print(f"[INFO] num_frames={num_frames}")

    # ----------------------------------------------------------
    # 3) Mapping TEMPLATE → CLASS_ID usando config.id2label
    # ----------------------------------------------------------
    id2label_cfg = model.config.id2label
    if not isinstance(id2label_cfg, dict) or len(id2label_cfg) == 0:
        print("[ERR] model.config.id2label vacío. Revisa el modelo HF.")
        sys.exit(1)

    template_to_id: Dict[str, int] = {}
    for k, v in id2label_cfg.items():
        try:
            idx = int(k)
        except Exception:
            idx = int(v) if str(v).isdigit() else None
        if idx is None:
            continue
        template_to_id[str(v)] = idx

    print(f"[INFO] Clases (templates) en modelo HF: {len(template_to_id)}")

    # ----------------------------------------------------------
    # 4) Cargar checkpoint entrenado (A o B)
    # ----------------------------------------------------------
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERR] No existe ckpt: {ckpt_path}")
        sys.exit(1)

    print(f"[LOAD] Checkpoint entrenado: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] missing_keys   : {len(missing)}")
    print(f"[INFO] unexpected_keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    # ----------------------------------------------------------
    # 5) Dataset + DataLoader
    # ----------------------------------------------------------
    dataset = SomethingSomethingV2Dataset(
        split=args.split,
        annotation_file=ann_file,
        video_root=video_root,
        template_to_id=template_to_id,
        num_frames=num_frames,
        test_answers_csv=test_answers_csv if args.split == "test" else None,
        max_videos=args.max_videos,
    )

    collate_fn = make_collate_fn(processor)

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # ----------------------------------------------------------
    # 6) Evaluación
    # ----------------------------------------------------------
    print("\n==========================================================")
    print(f"🔍 Evaluando SSv2 ({args.split}) con ckpt: {ckpt_path.name}")
    print("==========================================================\n")

    top1, top5, n = evaluate(model, dl, device)
    print("\n================ RESULTADOS (CKPT) =================")
    print(f"Split         : {args.split}")
    print(f"Vídeos eval   : {n}")
    print(f"Top-1 accuracy: {top1:.3f} %")
    print(f"Top-5 accuracy: {top5:.3f} %")
    print("====================================================\n")


if __name__ == "__main__":
    main()
