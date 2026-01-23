#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ssv2_attentive_probe.py

Probe atento tipo V-JEPA 2 sobre SSv2, con encoder congelado.

- Backbone: AutoModelForVideoClassification (facebook/vjepa2-vitl-fpc16-256-ssv2)
- Opción: encoder_override con BADAS (encoder.* en formato HF)
- Probe: AttentiveClassifier (4 bloques, num_heads=16) sobre tokens espacio-temporales.
- Métricas: Top-1 / Top-5 en validation.

Uso típico:

  # Modelo A (encoder HF original)
  CUDA_VISIBLE_DEVICES=0 python examples/train_ssv2_attentive_probe.py \
    --hf_repo facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --batch-size 8 \
    --num-workers 8 \
    --epochs 20 \
    --lr 3e-3 \
    --weight-decay 0.4 \
    --save-dir weights_ssv2_probe_A

  # Modelo B (encoder BADAS)
  CUDA_VISIBLE_DEVICES=1 python examples/train_ssv2_attentive_probe.py \
    --hf_repo facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --encoder_override weights/badas_encoder_for_ssv2_hf.pt \
    --batch-size 8 \
    --num-workers 8 \
    --epochs 20 \
    --lr 3e-3 \
    --weight-decay 0.4 \
    --save-dir weights_ssv2_probe_B
"""

import os
import sys
import json
import math
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

from torch.nn.init import trunc_normal_ as torch_trunc_normal_

# ----------------------------------------------------------------------
# MLP simple tipo ViT
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ----------------------------------------------------------------------
# Bloque Transformer estándar (self-attention + MLP)
# ----------------------------------------------------------------------
class Block(nn.Module):
    """
    Bloque tipo ViT:
      x -> LN -> MultiHeadSelfAttn -> +residual
         -> LN -> MLP -> +residual
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: bool = False,   # no lo usamos, lo dejamos por compatibilidad
        norm_layer=nn.LayerNorm,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=qkv_bias,
        )
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, drop=mlp_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        attn_out = self.proj_drop(attn_out)
        x = x + attn_out

        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


# ----------------------------------------------------------------------
# CrossAttention: queries Q contra tokens X (K/V)
# ----------------------------------------------------------------------
class CrossAttention(nn.Module):
    """
    Attention(Q, K=X, V=X).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: bool = False,  # no lo usamos
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.q_norm = norm_layer(dim)
        self.kv_norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=qkv_bias,
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        q: (B, Nq, D)
        x: (B, Nx, D)
        """
        qn = self.q_norm(q)
        xn = self.kv_norm(x)
        out, _ = self.attn(qn, xn, xn, need_weights=False)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ----------------------------------------------------------------------
# CrossAttentionBlock: cross-attn + MLP sobre los queries
# ----------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    Igual idea que Block, pero la atención es cross (Q contra X),
    y el MLP solo se aplica sobre Q.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, drop=mlp_drop)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        q: (B, Nq, D)
        x: (B, Nx, D)
        """
        # Cross-attn sobre Q
        q = q + self.cross_attn(q, x)
        # MLP sobre Q
        q = q + self.mlp(self.norm(q))
        return q


# ----------------------------------------------------------------------
# Pequeño wrapper para trunc_normal_
# ----------------------------------------------------------------------
def trunc_normal_(tensor, std: float = 0.02):
    return torch_trunc_normal_(tensor, mean=0.0, std=std)



# ----------------------------------------------------------------------
#  Attentive Probe (idéntico al paper, adaptado a D=1024, C=174)
# ----------------------------------------------------------------------
class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias
            )

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for _ in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                # attn: MultiheadAttention → out_proj
                if hasattr(layer.attn, "out_proj"):
                    rescale(layer.attn.out_proj.weight.data, layer_id + 1)
                # MLP: fc2
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        return q


class AttentiveClassifier(nn.Module):
    """Attentive Classifier"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.linear(x)
        return x

# ----------------------------------------------------------------------
#  Utils: localizar V-JEPA 2 y dataset SSv2
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
#  Dataset SSv2 (simple: 1 clip de 16 frames por vídeo)
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
        # IMPORTANTE: usar keyword videos=...
        inputs = processor(videos=list(videos_np), return_tensors="pt")
        labels_t = torch.tensor(labels, dtype=torch.long)
        return inputs, labels_t
    return collate_fn


def extract_pixel_values(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Intenta encontrar el tensor de vídeo dentro de `inputs`.
    Primero busca 'pixel_values'. Si no existe, busca cualquier tensor 5D
    y lo usa como fallback, imprimiendo las claves para debug.
    """
    if "pixel_values" in inputs:
        return inputs["pixel_values"]

    # Fallback heurístico: cualquier tensor 5D
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.ndim == 5:
            print(f"[WARN] 'pixel_values' no está en inputs. Usando inputs['{k}'] como vídeo.")
            print(f"[WARN] Claves completas de inputs: {list(inputs.keys())}")
            return v

    raise KeyError(f"No encuentro tensor de vídeo en inputs. Claves={list(inputs.keys())}")
# ----------------------------------------------------------------------
#  Modelo: backbone V-JEPA 2 (HF) + AttentiveClassifier
# ----------------------------------------------------------------------
class VJEPAAttentiveProbe(nn.Module):
    def __init__(self, hf_model: AutoModelForVideoClassification, num_classes: int = 174):
        super().__init__()
        self.encoder = hf_model.vjepa2
        self.probe = AttentiveClassifier(
            embed_dim=1024,
            num_heads=16,
            depth=4,
            num_classes=num_classes,
        )
        # congelar encoder por si acaso
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # encoder SIEMPRE en eval → sin dropout ni layernorm rara
        self.encoder.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, T, C, H, W)
        with torch.no_grad():
            # *** CLAVE: el encoder V-JEPA2 espera pixel_values_videos ***
            out = self.encoder(pixel_values_videos=pixel_values)

        # Intentar varias convenciones de salida, por robustez
        if hasattr(out, "last_hidden_state_videos"):
            tokens = out.last_hidden_state_videos      # (B, N, D)
        elif hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state             # (B, N, D)
        elif isinstance(out, tuple):
            tokens = out[0]
        else:
            raise RuntimeError(
                f"No encuentro last_hidden_state_videos / last_hidden_state en salida del encoder: {type(out)}"
            )

        logits = self.probe(tokens)  # (B, num_classes)
        return logits



# ----------------------------------------------------------------------
#  Entrenamiento / Validación
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
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    running_loss = 0.0
    correct_top1 = 0
    n_samples = 0

    it = dataloader
    if tqdm is not None:
        it = tqdm(dataloader, desc=f"Train (probe) E{epoch:03d}", unit="batch")

    for batch in it:
        inputs, labels = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
        #     pixel_values = extract_pixel_values(inputs)  # (B, T, C, H, W)
        #     logits = model(pixel_values)                # (B, C)
        #     loss = loss_fn(logits, labels)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            # --- coger el tensor de vídeo del processor ---
            if "pixel_values" in inputs:
                pixel_values = inputs["pixel_values"]              # caso genérico HF
            elif "pixel_values_videos" in inputs:
                # V-JEPA 2 usa esta clave
                pixel_values = inputs["pixel_values_videos"]
            else:
                raise KeyError(
                    f"No encuentro ni 'pixel_values' ni 'pixel_values_videos' en inputs. "
                    f"Claves = {list(inputs.keys())}"
                )

            logits = model(pixel_values)          # (B, C)
            loss = loss_fn(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

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
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    n_samples = 0

    it = dataloader
    if tqdm is not None:
        it = tqdm(dataloader, desc="Val (probe)", unit="batch")

    for batch in it:
        inputs, labels = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        labels = labels.to(device, non_blocking=True)

        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
        elif "pixel_values_videos" in inputs:
            pixel_values = inputs["pixel_values_videos"]
        else:
            raise KeyError(
                f"No encuentro ni 'pixel_values' ni 'pixel_values_videos' en inputs. "
                f"Claves = {list(inputs.keys())}"
            )

        logits = model(pixel_values)

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
#  MAIN
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_override", type=str, default="",
                    help="(opcional) encoder BADAS en formato HF (encoder.*).")
    ap.add_argument("--hf_repo", type=str,
                    default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.4)
    ap.add_argument("--save-dir", type=str, default="weights_ssv2_probe")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-train-videos", type=int, default=-1)
    ap.add_argument("--max-val-videos", type=int, default=-1)
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- localizar V-JEPA 2 root y SSv2 ---
    script_dir = Path(__file__).parent.resolve()
    vjepa_root = find_vjepa_root(script_dir)
    if vjepa_root is None:
        vjepa_root = Path.home() / "Desktop/Master Thesis/Experiments/V-JEPA 2"
        if not vjepa_root.exists():
            print("[ERR] No se encuentra 'V-JEPA 2'.")
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
            print(f"[ERR] No encuentro vídeos SSv2 en {video_root} ni {alt}")
            sys.exit(1)

    train_json = ssv2_meta_root / "train.json"
    val_json = ssv2_meta_root / "validation.json"
    print(f"[DATA] JSON train: {train_json}")
    print(f"[DATA] JSON val  : {val_json}")
    print(f"[DATA] Vídeos    : {video_root}")

    # --- device ---
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[INFO] Device: {device}")

    # --- modelo HF base ---
    print(f"[LOAD] Modelo HF: {args.hf_repo}")
    hf_model = AutoModelForVideoClassification.from_pretrained(args.hf_repo)
    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)

    num_frames = getattr(hf_model.config, "num_frames", None)
    if num_frames is None:
        num_frames = getattr(hf_model.config, "frames_per_clip", 16)
    print(f"[INFO] num_frames={num_frames}")

    # --- mapping template -> class id ---
    id2label_cfg = hf_model.config.id2label
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

    # --- inyectar encoder BADAS (opcional) ---
    if args.encoder_override:
        enc_path = Path(args.encoder_override)
        if not enc_path.exists():
            print(f"[ERR] encoder_override no existe: {enc_path}")
            sys.exit(1)

        print(f"[LOAD] BACKBONE BADAS HF: {enc_path}")
        enc_sd = torch.load(enc_path, map_location="cpu")
        hf_sd = hf_model.state_dict()
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

        hf_model.load_state_dict(new_sd, strict=False)
        print(f"[INFO] BADAS encoder.* candidatos: {n_cand}")
        print(f"[INFO] Aplicados: {n_applied}, missing: {n_missing}, shape mismatch: {n_shape}")
    else:
        print("[INFO] Sin encoder_override: encoder HF original (Modelo A).")

    # --- congelar TODO el modelo HF ---
    for p in hf_model.parameters():
        p.requires_grad = False

    # --- montar probe atento ---
    model = VJEPAAttentiveProbe(hf_model, num_classes=len(template_to_id))
    for p in model.probe.parameters():
        p.requires_grad = True
    model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OPT] Parámetros entrenables (sólo probe): {n_trainable:,}")

    # --- datasets / dataloaders ---
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

    # --- optimizer / loss ---
    optimizer = torch.optim.AdamW(
        model.probe.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- loop entrenamiento ---
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

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            best_epoch = epoch
            ckpt_path = save_dir / "ssv2_attentive_probe_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                    "encoder_override": args.encoder_override,
                },
                ckpt_path,
            )
            print(f"[CKPT] Nuevo mejor modelo guardado en: {ckpt_path} (ValAcc1={val_acc1:.2f}%)")

    print("\n================ RESUMEN ENTRENAMIENTO ================")
    print(f"Mejor época   : {best_epoch}")
    print(f"Mejor ValAcc1 : {best_acc1:.2f} %")
    print(f"Checkpoint    : {save_dir / 'ssv2_attentive_probe_best.pt'}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
