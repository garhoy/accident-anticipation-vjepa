#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ssv2_backbones_head_hf.py

Opción A: evaluar SSv2 con la CABEZA OFICIAL de V-JEPA2:

  Modelo A: V-JEPA2-SSv2 tal cual (encoder + head HF)
  Modelo B: Mismo head HF, pero encoder sobreescrito con BADAS
            (encoder.* en formato HF: badas_encoder_for_ssv2_hf.pt)

Mide Top-1 / Top-5 en SSv2 (split validation o train) sin entrenar nada.

Uso típico:

  # Modelo A (V-JEPA2 original)
  CUDA_VISIBLE_DEVICES=0 ~/.local/bin/poetry run python examples/eval_ssv2_backbones_head_hf.py \
    --hf_repo facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --split validation \
    --batch-size 8 \
    --num-workers 8

  # Modelo B (encoder BADAS)
  CUDA_VISIBLE_DEVICES=0 ~/.local/bin/poetry run python examples/eval_ssv2_backbones_head_hf.py \
    --hf_repo facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --encoder_override BADAS-Open/weights/badas_encoder_for_ssv2_hf.pt \
    --split validation \
    --batch-size 8 \
    --num-workers 8
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ----------------------------------------------------------------------
# Utils: localizar V-JEPA 2 y vídeos SSv2
# ----------------------------------------------------------------------

def find_vjepa_root(start_path: Path, max_levels: int = 5) -> Optional[Path]:
    current = start_path
    for _ in range(max_levels):
        candidate = current.parent / "V-JEPA 2"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    return None


def resolve_video_path(root: Path, vid: str) -> Optional[Path]:
    exts = [".webm", ".mp4", ".avi", ".mkv", ".MOV", ".mov"]
    for ext in exts:
        p = root / f"{vid}{ext}"
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------------
# Dataset SSv2 (mismo concepto que en tu train_ssv2_attentive_probe.py)
# ----------------------------------------------------------------------

class SomethingSomethingV2Dataset(Dataset):
    def __init__(
        self,
        split: str,
        annotation_file: Path,
        video_root: Path,
        template_to_id: Dict[str, int],
        num_frames: int,
        max_videos: int = -1,
    ):
        super().__init__()
        self.video_root = video_root
        self.template_to_id = template_to_id
        self.num_frames = num_frames

        with annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples: List[Tuple[str, int]] = []
        for row in data:
            vid = str(row.get("id", "")).strip()
            if not vid:
                continue
            template = str(row.get("template", "")).strip()
            if not template or template not in self.template_to_id:
                continue

            vpath = resolve_video_path(video_root, vid)
            if vpath is None:
                continue

            cls_id = int(self.template_to_id[template])
            samples.append((str(vpath), cls_id))

            if max_videos > 0 and len(samples) >= max_videos:
                break

        if not samples:
            raise RuntimeError(f"Dataset {split} vacío tras filtrar.")

        self.samples = samples
        print(f"[DATA] SSv2 ({split}): {len(self.samples)} vídeos listos ({annotation_file.name})")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, path: str):
        vframes, _, _ = read_video(path, pts_unit="sec")  # (T_total, H, W, C)
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
        return vframes.numpy()  # (T, H, W, C), uint8

    def __getitem__(self, idx: int):
        vpath, cls_id = self.samples[idx]
        video_np = self._load_clip(vpath)
        return video_np, cls_id


def make_collate_fn(processor: AutoVideoProcessor):
    def collate_fn(batch: List[Tuple[Any, int]]):
        videos_np, labels = zip(*batch)
        inputs = processor(videos=list(videos_np), return_tensors="pt")
        labels_t = torch.tensor(labels, dtype=torch.long)
        return inputs, labels_t
    return collate_fn


# ----------------------------------------------------------------------
# Evaluación con cabeza HF (sin entrenar)
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_hf_model(
    model: AutoModelForVideoClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    n_samples = 0

    it = dataloader
    if tqdm is not None:
        it = tqdm(dataloader, desc="Eval (HF head)", unit="batch")

    for batch in it:
        inputs, labels = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = loss_fn(logits, labels)

        running_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)

        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        correct_top1 += correct[:1].reshape(-1).float().sum().item()
        correct_top5 += correct[:maxk].reshape(-1).float().sum().item()

        if tqdm is not None:
            avg_loss = running_loss / max(1, n_samples)
            acc1 = 100.0 * correct_top1 / max(1, n_samples)
            acc5 = 100.0 * correct_top5 / max(1, n_samples)
            it.set_postfix(loss=f"{avg_loss:.4f}", acc1=f"{acc1:.2f}%", acc5=f"{acc5:.2f}%")

    avg_loss = running_loss / max(1, n_samples)
    acc1 = 100.0 * correct_top1 / max(1, n_samples)
    acc5 = 100.0 * correct_top5 / max(1, n_samples)
    return avg_loss, acc1, acc5


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_repo", type=str, default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--encoder_override", type=str, default="",
                    help="(opcional) encoder BADAS en formato HF (encoder.*).")
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    script_dir = Path(__file__).parent.resolve()
    vjepa_root = find_vjepa_root(script_dir)
    if vjepa_root is None:
        vjepa_root = Path.home() / "Desktop/Master Thesis/Experiments/V-JEPA 2"
        if not vjepa_root.exists():
            print("[ERR] No se encuentra 'V-JEPA 2'.")
            sys.exit(1)
    print(f"[OK] V-JEPA 2 root: {vjepa_root}")

    # Rutas SSv2
    ssv2_raw_root = vjepa_root / "data/raw/SSv2"
    ssv2_meta_root = vjepa_root / "data/metadata/SSv2/labels"
    video_root = ssv2_raw_root / "20bn-something-something-v2"
    if not video_root.exists():
        alt = ssv2_raw_root / "Videos"
        if alt.exists():
            video_root = alt
        else:
            print(f"[ERR] No encuentro vídeos SSv2 en {video_root} ni {alt}")
            sys.exit(1)

    if args.split == "train":
        ann_file = ssv2_meta_root / "train.json"
    else:
        ann_file = ssv2_meta_root / "validation.json"

    print(f"[DATA] JSON {args.split}: {ann_file}")
    print(f"[DATA] Vídeos        : {video_root}")

    # Device
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[INFO] Device: {device}")

    # Modelo HF
    print(f"[LOAD] Modelo HF: {args.hf_repo}")
    model = AutoModelForVideoClassification.from_pretrained(args.hf_repo)
    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)

    num_frames = getattr(model.config, "num_frames", None)
    if num_frames is None:
        num_frames = getattr(model.config, "frames_per_clip", 16)
    print(f"[INFO] num_frames={num_frames}")

    # Mapping template -> class id (igual que en tu script)
    id2label_cfg = model.config.id2label
    template_to_id: Dict[str, int] = {}
    for k, v in id2label_cfg.items():
        try:
            idx = int(k)
        except Exception:
            idx = int(v) if str(v).isdigit() else None
        if idx is None:
            continue
        template_to_id[str(v)] = idx
    print(f"[INFO] Clases HF: {len(template_to_id)}")

    # OVERWRITE encoder con BADAS (opcional)
    if args.encoder_override:
        enc_path = Path(args.encoder_override)
        if not enc_path.exists():
            print(f"[ERR] encoder_override no existe: {enc_path}")
            sys.exit(1)

        print(f"[LOAD] BACKBONE BADAS HF: {enc_path}")
        enc_sd = torch.load(enc_path, map_location="cpu")
        hf_sd = model.state_dict()
        new_sd = hf_sd.copy()

        n_cand = n_applied = n_missing = n_shape = 0
        for k_badas, v_badas in enc_sd.items():
            if not k_badas.startswith("encoder."):
                continue
            n_cand += 1
            k_hf = "vjepa2." + k_badas
            if k_hf not in new_sd:
                n_missing += 1
                continue
            if new_sd[k_hf].shape != v_badas.shape:
                n_shape += 1
                continue
            new_sd[k_hf] = v_badas
            n_applied += 1

        model.load_state_dict(new_sd, strict=False)
        print(f"[INFO] BADAS encoder.* candidatos: {n_cand}")
        print(f"[INFO] Aplicados: {n_applied}, missing: {n_missing}, shape mismatch: {n_shape}")
    else:
        print("[INFO] Sin encoder_override: V-JEPA2 original (Modelo A).")

    model.to(device)

    # Dataset + loader
    train_ds = SomethingSomethingV2Dataset(
        split=args.split,
        annotation_file=ann_file,
        video_root=video_root,
        template_to_id=template_to_id,
        num_frames=num_frames,
        max_videos=args.max_videos,
    )

    collate_fn = make_collate_fn(processor)
    ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Eval
    loss, acc1, acc5 = evaluate_hf_model(model, ld, device)

    tag = "VJEPA2" if not args.encoder_override else "BADAS-ENC"
    print("\n================ RESULTADOS =================")
    print(f"Modelo         : {tag}")
    print(f"Split          : {args.split}")
    print(f"#videos        : {len(train_ds)}")
    print(f"Loss           : {loss:.4f}")
    print(f"Top-1 accuracy : {acc1:.2f} %")
    print(f"Top-5 accuracy : {acc5:.2f} %")
    print("================================================\n")


if __name__ == "__main__":
    main()
