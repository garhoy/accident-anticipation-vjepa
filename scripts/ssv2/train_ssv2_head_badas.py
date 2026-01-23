#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ssv2_head_badas.py

Entrena SOLO la cabeza de clasificación de SSv2 usando:

  - Backbone: V-JEPA 2 ViT-L (facebook/vjepa2-vitl-fpc16-256-ssv2)
  - Encoder sobreescrito con BADAS (encoder_override .pt en formato HF)
  - Cabeza: Linear(d_model -> 174) reinicializada desde cero

Importante:
  - El encoder (BACKBONE) se congela completamente (requires_grad=False).
  - SOLO se entrenan los pesos de la cabeza (classifier).
  - Pérdida: CrossEntropyLoss.
  - Dataset: Something-Something V2 (train.json / validation.json).
  - Opcional: submuestreo con --max-train-videos / --max-val-videos.

Uso típico:

  python examples/train_ssv2_head_badas.py \
    --encoder_override weights/badas_encoder_for_ssv2_hf.pt \
    --hf_repo facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --batch-size 8 \
    --num-workers 8 \
    --epochs 10 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --max-train-videos 50000 \
    --max-val-videos 5000 \
    --save-dir weights_ssv2_badas_head
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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
#  Utils: localizar V-JEPA 2 y resolver rutas
# ----------------------------------------------------------------------

def find_vjepa_root(start_path: Path, max_levels: int = 5) -> Optional[Path]:
    """
    Sube directorios hasta encontrar 'V-JEPA 2'.
    Igual que en eval_badas.py.
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
    Para split 'train' / 'val': usa annotation_file (train.json / validation.json)
      - Cada entrada tiene: id, label, template, placeholders
      - Usamos SOLO 'template' para mapear a la clase (como en HF).

    Para este script solo usamos train/val, pero dejamos 'test' implementado.
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
            # Dejado por coherencia; no lo usamos aquí.
            if test_answers_csv is None or not test_answers_csv.exists():
                raise FileNotFoundError(
                    f"Se necesita test-answers.csv para el split 'test': {test_answers_csv}"
                )
            id2template: Dict[str, str] = {}
            with test_answers_csv.open("r", encoding="utf-8") as fcsv:
                for line in fcsv:
                    line = line.strip()
                    if not line:
                        continue
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
#  Train / Val loops (SOLO HEAD ENTRENABLE)
# ----------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Entrena una época completa (solo head tiene grad).
    Devuelve: (loss_medio, top1_acc)
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    running_loss = 0.0
    correct_top1 = 0
    n_samples = 0

    it = dataloader
    if tqdm is not None:
        it = tqdm(dataloader, desc=f"Train (head) E{epoch:03d}", unit="batch")

    for batch in it:
        inputs, labels = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(**inputs)
            logits = outputs.logits  # (B, num_classes)
            loss = loss_fn(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # stats
        running_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()

        if tqdm is not None:
            avg_loss = running_loss / max(1, n_samples)
            acc1 = 100.0 * correct_top1 / max(1, n_samples)
            it.set_postfix(loss=f"{avg_loss:.4f}", acc1=f"{acc1:.2f}%")

    avg_loss = running_loss / max(1, n_samples)
    acc1 = 100.0 * correct_top1 / max(1, n_samples)
    return avg_loss, acc1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evalúa en val:
      - loss medio (CrossEntropy)
      - top-1 accuracy (%)
      - top-5 accuracy (%)
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    n_samples = 0

    it = dataloader
    if tqdm is not None:
        it = tqdm(dataloader, desc="Val (head)", unit="batch")

    for batch in it:
        inputs, labels = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        outputs = model(**inputs)
        logits = outputs.logits  # (B, num_classes)
        loss = loss_fn(logits, labels)

        running_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)

        # top-1 / top-5
        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
        pred = pred.t()  # (maxk, B)
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # (maxk, B)

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
#  MAIN
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoder_override",
        type=str,
        default="",
        help="(opcional) Ruta a encoder BADAS en formato HF. Si se omite, se usa el encoder original de V-JEPA2.",
    )

    ap.add_argument(
        "--hf_repo",
        type=str,
        default="facebook/vjepa2-vitl-fpc16-256-ssv2",
        help="Repo HF del modelo SSv2 oficial de V-JEPA 2.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size de entrenamiento.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Num workers del DataLoader.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas de entrenamiento.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate para la cabeza.",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay del optimizador.",
    )
    ap.add_argument(
        "--save-dir",
        type=str,
        default="weights_ssv2_badas_head",
        help="Directorio para guardar checkpoints.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' o 'cpu'. Por defecto usa CUDA si está disponible.",
    )
    ap.add_argument(
        "--max-train-videos",
        type=int,
        default=-1,
        help="Máximo nº de vídeos de train (<=0 = todos). Útil para reducir coste.",
    )
    ap.add_argument(
        "--max-val-videos",
        type=int,
        default=-1,
        help="Máximo nº de vídeos de val (<=0 = todos).",
    )
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    train_json = ssv2_meta_root / "train.json"
    val_json = ssv2_meta_root / "validation.json"

    print(f"[DATA] JSON train : {train_json}")
    print(f"[DATA] JSON val   : {val_json}")
    print(f"[DATA] Vídeos raíz: {video_root}")

    # ----------------------------------------------------------
    # 1) Device
    # ----------------------------------------------------------
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[INFO] Device: {device}")

    # ----------------------------------------------------------
    # 2) Cargar modelo SSv2 oficial (base)
    # ----------------------------------------------------------
    print(f"[LOAD] Modelo HF: {args.hf_repo}")
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
    # 4) Inyectar BACKBONE BADAS en vjepa2.encoder.*
    # ----------------------------------------------------------
    # enc_path = Path(args.encoder_override)
    # if not enc_path.exists():
    #     print(f"[ERR] No existe encoder_override: {enc_path}")
    #     sys.exit(1)

    # print(f"[LOAD] BACKBONE BADAS HF: {enc_path}")
    # enc_sd = torch.load(enc_path, map_location="cpu")

    # # Debug: primeras claves
    # print("\n[DEBUG] Primeras claves del modelo HF:")
    # hf_sd = model.state_dict()
    # for i, k in enumerate(hf_sd.keys()):
    #     if i >= 20:
    #         break
    #     print(f"  HF: {k}")

    # print("\n[DEBUG] Primeras claves del encoder BADAS:")
    # for i, k in enumerate(enc_sd.keys()):
    #     if i >= 20:
    #         break
    #     print(f"  BADAS: {k}")

    # # Remap encoder.* → vjepa2.encoder.*
    # new_sd = hf_sd.copy()
    # n_cand = 0
    # n_applied = 0
    # n_missing = 0
    # n_shape = 0

    # for k_badas, v_badas in enc_sd.items():
    #     if not k_badas.startswith("encoder."):
    #         continue
    #     n_cand += 1
    #     k_hf = "vjepa2." + k_badas
    #     if k_hf not in new_sd:
    #         n_missing += 1
    #         continue
    #     if new_sd[k_hf].shape != v_badas.shape:
    #         n_shape += 1
    #         continue
    #     new_sd[k_hf] = v_badas
    #     n_applied += 1

    # model.load_state_dict(new_sd, strict=False)

    # print(f"[INFO] Pesos BADAS candidatos (encoder.*): {n_cand}")
    # print(f"[INFO] Pesos remapeados y aplicados al encoder: {n_applied}")
    # print(f"[INFO] Skips por missing key: {n_missing} | por shape mismatch: {n_shape}")
    # ----------------------------------------------------------
    # 4) Inyectar BACKBONE BADAS en vjepa2.encoder.* (opcional)
    # ----------------------------------------------------------
    if args.encoder_override:
        enc_path = Path(args.encoder_override)
        if not enc_path.exists():
            print(f"[ERR] No existe encoder_override: {enc_path}")
            sys.exit(1)

        print(f"[LOAD] BACKBONE BADAS HF: {enc_path}")
        enc_sd = torch.load(enc_path, map_location="cpu")

        # Debug: primeras claves
        print("\n[DEBUG] Primeras claves del modelo HF:")
        hf_sd = model.state_dict()
        for i, k in enumerate(hf_sd.keys()):
            if i >= 20:
                break
            print(f"  HF: {k}")

        print("\n[DEBUG] Primeras claves del encoder BADAS:")
        for i, k in enumerate(enc_sd.keys()):
            if i >= 20:
                break
            print(f"  BADAS: {k}")

        new_sd = hf_sd.copy()
        n_cand = 0
        n_applied = 0
        n_missing = 0
        n_shape = 0

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

        print(f"[INFO] Pesos BADAS candidatos (encoder.*): {n_cand}")
        print(f"[INFO] Pesos remapeados y aplicados al encoder: {n_applied}")
        print(f"[INFO] Skips por missing key: {n_missing} | por shape mismatch: {n_shape}")
    else:
        print("[INFO] Sin encoder_override: usando encoder ORIGINAL de V-JEPA2 (Modelo A).")

    # ----------------------------------------------------------
    # 5) Reinicializar head y CONGELAR encoder
    # ----------------------------------------------------------
    # Congelar TODO primero
    for p in model.parameters():
        p.requires_grad = False

    # Reinicializar classifier
    if not hasattr(model, "classifier"):
        print("[ERR] El modelo HF no tiene atributo 'classifier'. Revisa la arquitectura.")
        sys.exit(1)

    in_dim = model.classifier.in_features
    num_classes = model.config.num_labels
    model.classifier = nn.Linear(in_dim, num_classes)

    # Activar grad SOLO en la head
    for p in model.classifier.parameters():
        p.requires_grad = True

    print(f"[HEAD] Cabeza reinicializada: Linear({in_dim} -> {num_classes})")

    # Comprobar nº de parámetros entrenables
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OPT] Parámetros entrenables (solo head): {n_trainable:,}")

    model.to(device)

    # ----------------------------------------------------------
    # 6) Datasets + Dataloaders
    # ----------------------------------------------------------
    train_ds = SomethingSomethingV2Dataset(
        split="train",
        annotation_file=train_json,
        video_root=video_root,
        template_to_id=template_to_id,
        num_frames=num_frames,
        max_videos=args.max_train_videos,
    )

    val_ds = SomethingSomethingV2Dataset(
        split="val",
        annotation_file=val_json,
        video_root=video_root,
        template_to_id=template_to_id,
        num_frames=num_frames,
        max_videos=args.max_val_videos,
    )

    collate_fn = make_collate_fn(processor)

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # ----------------------------------------------------------
    # 7) Optimizer (SOLO head)
    # ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    # ----------------------------------------------------------
    # 8) Loop de entrenamiento con best checkpoint por Top-1 val
    # ----------------------------------------------------------
    best_acc1 = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Época {epoch:03d}/{args.epochs:03d} ==========")

        train_loss, train_acc1 = train_one_epoch(
            model=model,
            dataloader=train_ld,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
            use_amp=True,
        )

        val_loss, val_acc1, val_acc5 = evaluate(
            model=model,
            dataloader=val_ld,
            device=device,
        )

        print(
            f"[E{epoch:03d}] "
            f"TrainLoss={train_loss:.4f} | TrainAcc1={train_acc1:.2f}% | "
            f"ValLoss={val_loss:.4f} | ValAcc1={val_acc1:.2f}% | ValAcc5={val_acc5:.2f}%"
        )

        # Guardar mejor head según Top-1 val
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            best_epoch = epoch
            ckpt_path = save_dir / "badas_ssv2_head_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),  # incluye encoder BADAS congelado + head entrenada
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                },
                ckpt_path,
            )
            print(f"[CKPT] Nuevo mejor modelo guardado en: {ckpt_path} (ValAcc1={val_acc1:.2f}%)")

    print("\n================ RESUMEN ENTRENAMIENTO ================")
    print(f"Mejor época   : {best_epoch}")
    print(f"Mejor ValAcc1 : {best_acc1:.2f} %")
    print(f"Checkpoint    : {save_dir / 'badas_ssv2_head_best.pt'}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
