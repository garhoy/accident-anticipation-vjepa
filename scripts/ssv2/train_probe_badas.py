# train_probe_badas.py
# Linear probing con AttentiveClassifier (4 capas) sobre BADAS congelado.
# Soporta SSv2 (última capa) y Jester (capas [17,19,21,23]) como en el paper.

import os, csv, math, time, json, argparse, random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from badas_adapter import BADASBackbone
from attentive_head import AttentiveClassifier


# ========================= Config por dataset (ViT-L) =========================
DATASET_CFGS: Dict[str, Dict] = {
    # Something-Something v2 (SSv2)
    "ssv2": {
        "frames": 16, "segments": 2, "views": 3, "frame_step": 4,
        "epochs": 20, "global_bsz": 256,
        "lr_grid": [5e-3, 3e-3, 1e-3, 3e-4, 1e-4],
        "wd_grid": [0.8, 0.4, 0.1, 0.01],
        "num_classes": 174,
        "encoder_layers": None,                 # última capa
        "attn_heads": 16, "probe_depth": 4,     # 4 capas (3 self + 1 cross)
    },
    # Jester (capas intermedias)
    "jester": {
        "frames": 32, "segments": 4, "views": 3, "frame_step": 2,
        "epochs": 100, "global_bsz": 128,
        "lr_grid": [1e-3, 3e-4, 1e-4],
        "wd_grid": [0.8],
        "num_classes": 27,
        "encoder_layers": [17, 19, 21, 23],     # ViT-L (24 bloques)
        "attn_heads": 16, "probe_depth": 4,
    },
}


# ================================ Utilidades ==================================
def set_seed(sd: int = 42):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)


def three_spatial_crops(x: torch.Tensor, size: int = 256) -> List[torch.Tensor]:
    """
    x: (T,C,H,W) -> devuelve 3 crops (left/center/right o top/center/bottom).
    """
    T, C, H, W = x.shape
    if H >= W:
        top = x[..., 0:size, :]
        c0 = (H - size) // 2
        center = x[..., c0:c0+size, :]
        bottom = x[..., H-size:H, :]
        return [top, center, bottom]
    else:
        left = x[..., :, 0:size]
        c0 = (W - size) // 2
        center = x[..., :, c0:c0+size]
        right = x[..., :, W-size:W]
        return [left, center, right]


def resize_shorter_to(x: torch.Tensor, short=256) -> torch.Tensor:
    """
    Reescala para que min(H,W)=short manteniendo aspecto y hace center-crop a short×short.
    x: (T,C,H,W)
    """
    import torch.nn.functional as F
    T, C, H, W = x.shape
    if H == W == short:
        return x
    scale = short / min(H, W)
    H2, W2 = int(round(H * scale)), int(round(W * scale))
    x = F.interpolate(x.float(), size=(H2, W2), mode="bilinear", align_corners=False)
    y0 = (H2 - short) // 2; x0 = (W2 - short) // 2
    return x[..., y0:y0+short, x0:x0+short]


def normalize_im(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1,3,1,1)
    return (x.clamp(0,1) - mean) / std


def npy_or_pt_to_tensor(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix.lower() == ".pt":
        t = torch.load(str(p), map_location="cpu")
        return t.float()
    elif p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        return torch.from_numpy(arr).float()
    raise ValueError(f"Extensión no soportada para clip tensor: {p.suffix}")


def decode_video(path: str) -> torch.Tensor:
    """
    Devuelve frames normalizados [T,C,H,W] en [0,1].
    Intenta primero decord; si falla, usa torchvision.io.read_video.
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path)
        frames = vr.get_batch(list(range(len(vr))))   # (T,H,W,3), uint8
        frames = frames.permute(0,3,1,2).float() / 255.0
        return frames
    except Exception:
        from torchvision.io import read_video
        vid, _, _ = read_video(path, output_format="TCHW")  # (T,C,H,W), uint8
        return vid.float() / 255.0


def temporal_segments_indices(T_total: int, frames: int, segments: int, step: int, train: bool) -> List[int]:
    """
    Devuelve índices de inicio para cada segmento; cada clip toma 'frames' con stride 'step'.
    """
    seg_span = max(1, (T_total - frames * step) // max(1, segments) + 1)
    starts = []
    for s in range(segments):
        lo = s * seg_span
        hi = min(lo + max(1, seg_span - 1), max(0, T_total - frames * step))
        if train:
            st = random.randint(lo, max(lo, hi))
        else:
            st = (lo + hi) // 2
        starts.append(st)
    return starts


# ================================= Dataset ====================================
class ClipsCSV(Dataset):
    """
    CSV con columnas: id, path, label
    - Si path .pt/.npy: tensor [T,C,H,W] (se normaliza/redimensiona igualmente).
    - Si path vídeo: decodifica y samplea segments × frames con frame_step.
    Train: 1 clip por item (concatenando segmentos por tiempo).
    Val: 3 crops por segmento (devuelve lista de vistas).
    """
    def __init__(self, csv_path: str, frames: int, segments: int, frame_step: int, mode: str):
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            need = {"id", "path", "label"}
            if not need.issubset(set(r.fieldnames or [])):
                raise ValueError(f"{csv_path} debe tener columnas {need}")
            for row in r:
                self.rows.append({"id": row["id"], "path": row["path"], "label": int(row["label"])})
        self.frames = frames
        self.segments = segments
        self.frame_step = frame_step
        self.mode = mode  # "train" | "val" | "test"

    def __len__(self): return len(self.rows)

    def _load_clip_tensor(self, path: str) -> torch.Tensor:
        p = Path(path)
        if p.suffix.lower() in {".pt", ".npy"}:
            return npy_or_pt_to_tensor(path)
        else:
            return decode_video(path)

    def __getitem__(self, idx):
        it = self.rows[idx]
        vid = self._load_clip_tensor(it["path"])  # (T0,C,H,W)
        assert vid.ndim == 4 and vid.shape[1] == 3, f"Esperaba [T,C,H,W] con C=3, obtuve {tuple(vid.shape)}"
        T0 = vid.shape[0]

        # Resize + normalización por frame
        vid = torch.stack([normalize_im(resize_shorter_to(f.unsqueeze(0), 256).squeeze(0)) for f in vid], dim=0)

        starts = temporal_segments_indices(T0, self.frames, self.segments, self.frame_step, train=(self.mode=="train"))
        clips = []
        for st in starts:
            idxs = [min(T0-1, max(0, st + i * self.frame_step)) for i in range(self.frames)]
            clip = vid[idxs]  # [T,3,256,256]
            clips.append(clip)

        label = it["label"]
        if self.mode == "train":
            # Aug simple: flip horizontal
            if random.random() < 0.5:
                clips = [c.flip(-1) for c in clips]
            # Concatenamos segmentos en el eje temporal
            return torch.cat(clips, dim=0), label  # [segments*T, 3, 256, 256]
        else:
            views = []
            for c in clips:
                for v in three_spatial_crops(c, size=256):
                    views.append(v)  # [T,3,256,256]
            return views, label


def collate_train(batch):
    clips = [b[0] for b in batch]  # [Ttot,3,256,256]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    Ttot = clips[0].shape[0]
    clips = torch.stack(clips, dim=0)  # [B, Ttot, 3, 256, 256]
    return {"video": clips, "label": labels}


def collate_eval(batch):
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    views = [b[0] for b in batch]  # list of lists (por item)
    return {"views": views, "label": labels}


# ========================= Runner multi-probe (grid) ===========================
class MultiProbeRunner:
    def __init__(self, embed_dim: int, num_classes: int, lr_list: List[float], wd_list: List[float],
                 depth: int = 4, heads: int = 16, device: str = "cuda",
                 warmup_epochs: int = 2, total_epochs: int = 20):
        self.device = device
        self.cfgs = [{"lr": lr, "wd": wd} for lr in lr_list for wd in wd_list]
        self.models: List[nn.Module] = []
        self.optims: List[optim.Optimizer] = []
        self.schedulers_warm: List[optim.lr_scheduler._LRScheduler] = []
        self.schedulers_main: List[optim.lr_scheduler._LRScheduler] = []
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.crit = nn.CrossEntropyLoss()

        for cfg in self.cfgs:
            m = AttentiveClassifier(
                embed_dim=embed_dim, num_heads=heads, mlp_ratio=4.0, depth=depth,
                num_classes=num_classes, complete_block=True
            ).to(device)
            self.models.append(m)
            opt = optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            self.optims.append(opt)
            # Warmup linear + Cosine
            self.schedulers_warm.append(optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warmup_epochs))
            self.schedulers_main.append(optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(total_epochs - warmup_epochs)))

    def step_schedulers(self, epoch: int):
        for sw, sm in zip(self.schedulers_warm, self.schedulers_main):
            if epoch <= self.warmup_epochs:
                sw.step()
            else:
                sm.step()

    def train_step_all_heads(self, toks: torch.Tensor, labels: torch.Tensor):
        for m, opt in zip(self.models, self.optims):
            m.train()
            opt.zero_grad(set_to_none=True)
            logits = m(toks)
            loss = self.crit(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

    @torch.no_grad()
    def eval_views(self, encoder: BADASBackbone, batch_views: List[List[torch.Tensor]], layers=None) -> List[torch.Tensor]:
        """
        Para cada modelo, devuelve logits promediados sobre (segments × 3 crops) por item.
        """
        outs = []
        for m in self.models:
            m.eval()
            all_logits = []
            for views in batch_views:  # vistas de un ítem
                logits_v = []
                for v in views:
                    v = v.unsqueeze(0).to(self.device)  # [1,T,3,256,256]
                    toks = encoder.encode_tokens(v, layers=layers)  # [1,N,D]
                    logit = m(toks)                                # [1,C]
                    logits_v.append(logit)
                all_logits.append(torch.stack(logits_v, dim=0).mean(dim=0))  # [1,C]
            outs.append(torch.cat(all_logits, dim=0))  # [B,C]
        return outs


# =================================== Main =====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CFGS.keys()))
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  default=None)
    ap.add_argument("--badas_ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_device_bsz", type=int, default=32)
    ap.add_argument("--accum_steps", type=int, default=1, help="para alcanzar el batch global del paper")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_dir", default="vjepa2_probe_on_badas")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = DATASET_CFGS[args.dataset]
    os.makedirs(args.save_dir, exist_ok=True)

    # Backbone BADAS congelado
    encoder = BADASBackbone(checkpoint=args.badas_ckpt, device=args.device, proj_to=1024)
    encoder.eval()

    # Datasets / loaders
    train_ds = ClipsCSV(args.train_csv, frames=cfg["frames"], segments=cfg["segments"],
                        frame_step=cfg["frame_step"], mode="train")
    val_ds   = ClipsCSV(args.val_csv,   frames=cfg["frames"], segments=cfg["segments"],
                        frame_step=cfg["frame_step"], mode="val")

    train_loader = DataLoader(train_ds, batch_size=args.per_device_bsz, shuffle=True,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate_train, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.per_device_bsz, shuffle=False,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate_eval)

    # Descubre D (por si BADAS cambia) con un batch
    with torch.no_grad():
        sample = next(iter(train_loader))
        toks0 = encoder.encode_tokens(sample["video"].to(args.device), layers=cfg["encoder_layers"])
        D = int(toks0.shape[-1])

    runner = MultiProbeRunner(
        embed_dim=D, num_classes=cfg["num_classes"],
        lr_list=cfg["lr_grid"], wd_list=cfg["wd_grid"],
        depth=cfg["probe_depth"], heads=cfg["attn_heads"],
        device=args.device, warmup_epochs=2, total_epochs=cfg["epochs"]
    )

    # Info
    eff_bsz = args.per_device_bsz * args.accum_steps
    print(f"[INFO] Dataset={args.dataset} | frames={cfg['frames']} segs={cfg['segments']} views={cfg['views']} step={cfg['frame_step']}")
    print(f"[INFO] Global batch target={cfg['global_bsz']} | efectivo≈{eff_bsz}")
    print(f"[INFO] LR grid={cfg['lr_grid']} | WD grid={cfg['wd_grid']}")
    print(f"[INFO] Encoder layers={cfg['encoder_layers'] if cfg['encoder_layers'] is not None else 'last'} | D={D}")

    best = {"acc": -1.0, "idx": -1, "epoch": -1, "lr": None, "wd": None}
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        # ------------------- TRAIN -------------------
        acc_correct = 0
        acc_total = 0
        for batch in train_loader:
            video = batch["video"].to(args.device)  # [B, Ttot, 3, 256, 256]
            labels = batch["label"].to(args.device)

            with torch.no_grad():
                toks = encoder.encode_tokens(video, layers=cfg["encoder_layers"])  # [B,N,D]

            # Entrenamos TODAS las cabezas del grid en paralelo
            runner.train_step_all_heads(toks, labels)

            # métrica rápida con la 1ª cabeza
            with torch.no_grad():
                m0 = runner.models[0].eval()
                logits0 = m0(toks)
                acc_correct += (logits0.argmax(dim=-1) == labels).sum().item()
                acc_total += labels.numel()

        runner.step_schedulers(ep)
        tr_acc = 100.0 * acc_correct / max(1, acc_total)
        print(f"[EP {ep:03d}] train_acc(head0)={tr_acc:.2f}%  time={time.time()-t0:.1f}s")

        # -------------------- VAL --------------------
        all_accs = [0.0 for _ in runner.models]
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["label"].to(args.device)
                logits_list = runner.eval_views(encoder, batch["views"], layers=cfg["encoder_layers"])
                for i, logits in enumerate(logits_list):
                    preds = logits.argmax(dim=-1)
                    all_accs[i] += (preds == labels).sum().item()
                total += labels.numel()

        val_accs = [a / max(1, total) for a in all_accs]
        best_i = int(np.argmax(val_accs))
        if val_accs[best_i] > best["acc"]:
            best.update({
                "acc": float(val_accs[best_i]),
                "idx": best_i,
                "epoch": ep,
                "lr": runner.cfgs[best_i]["lr"],
                "wd": runner.cfgs[best_i]["wd"],
            })
            out_ckpt = Path(args.save_dir) / f"{args.dataset}_best_probe.pt"
            torch.save({
                "epoch": ep,
                "acc": best["acc"],
                "probe_state_dict": runner.models[best_i].state_dict(),
                "embed_dim": D,
                "dataset": args.dataset,
                "cfg": cfg,
                "lr": best["lr"], "wd": best["wd"],
            }, out_ckpt)

        print(f"[EP {ep:03d}] val_acc_best={100*val_accs[best_i]:.2f}% (head#{best_i}, lr={runner.cfgs[best_i]['lr']}, wd={runner.cfgs[best_i]['wd']})")
        print(f"           all={ [round(100*a,2) for a in val_accs] } | best_so_far={round(100*best['acc'],2)}%@ep{best['epoch']}")

    out_json = Path(args.save_dir) / f"{args.dataset}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print("\n===== DONE =====")
    print(json.dumps(best, indent=2))
    print(f"Guardados: {out_json} y {args.dataset}_best_probe.pt")


if __name__ == "__main__":
    main()
