#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def _to_list_dirs(d: Union[str, Path, List[str], List[Path]]) -> List[Path]:
    if isinstance(d, (str, Path)):
        return [Path(d)]
    return [Path(x) for x in d]

class VJEPAFramewiseDataset(Dataset):
    """
    Secuencia por vÃƒÂ­deo (10 Hz, stride=1). Usa FIN de clip como timestamp base.
    Las etiquetas multi-horizonte son frame-wise: y[t,h]=1 si delta_end[t] <= -h en positivos.
    """
    def __init__(self,
                 embeddings_dirs: Union[str, Path, List[str], List[Path]],
                 window_s: float = 5.0,
                 post_window_s: float = 0.0,
                 anticipation_windows: List[float] = [1.0, 2.0, 3.0, 5.0],
                 video_ids: Optional[List[str]] = None):
        self.dirs = _to_list_dirs(embeddings_dirs)
        self.W = float(window_s)
        self.post = float(post_window_s)
        self.horizons = [float(x) for x in anticipation_windows]

        self.index: Dict[str, Path] = {}
        for d in self.dirs:
            if not d.exists(): 
                continue
            for p in d.glob("*.pt"):
                self.index[str(p)] = p

        self.ids = sorted(self.index.keys()) if video_ids is None else list(video_ids)

        self.meta = []
        for vid in self.ids:
            p = self.index.get(vid) or (self.index.get(vid.split("-", 1)[1]) if "-" in vid else None)
            if p is None:
                continue
            try:
                data = torch.load(p, map_location="cpu")
                z = data.get("z_clip", None)
                if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[0] == 0:
                    continue
                tgt = int(data.get("meta", {}).get("target", 0))
                self.meta.append({"id": str(data.get("id", vid)), "path": p, "target": tgt})
            except Exception as e:
                print(f"[WARN] {vid}: {e}")

        print(f"Dataset: {len(self.meta)}/{len(self.ids)} vÃƒÂ­deos vÃƒÂ¡lidos")
        if self.meta:
            npos = sum(m["target"] for m in self.meta)
            print(f"  Positivos: {npos} ({100*npos/len(self.meta):.1f}%) | Negativos: {len(self.meta)-npos}")

    def __len__(self):
        return len(self.meta)

    def get_sampler_weights(self) -> torch.Tensor:
        """Pesos para WeightedRandomSampler (balanceo 50/50)"""
        targets = torch.tensor([m["target"] for m in self.meta], dtype=torch.float32)
        n_pos = (targets == 1).sum().item()
        n_neg = (targets == 0).sum().item()
        
        w_pos = 0.5 / max(n_pos, 1)
        w_neg = 0.5 / max(n_neg, 1)
        
        weights = torch.where(targets == 1, w_pos, w_neg)
        return weights

    def __getitem__(self, idx):
        m = self.meta[idx]
        data = torch.load(m["path"], map_location="cpu")
        vid = m["id"]
        target = float(int(m["target"]))

        Zc: torch.Tensor = data["z_clip"].float()            # (T_all, D)
        Tc: torch.Tensor = data["clip_times"].float()        # (T_all,) centros
        Dc: torch.Tensor = data["delta_to_anchor"].float()   # (T_all,)
        meta_dict = data.get("meta", {})

        fps = float(data.get("fps_target", meta_dict.get("fps_target", 10.0)))
        clip_len = int(meta_dict.get("clip_len", 16))
        half = (clip_len - 1) / (2.0 * max(fps, 1e-6))  # 0.75 s para L=16@10Hz

        # FIN de clip = centro + half_span
        Te = Tc + half
        De = Dc + half


        # Recorte a [-W, post]
        mask = (De >= -self.W) & (De <= self.post)
        if not mask.any():
            # fallback: toma el frame mÃƒÂ¡s cercano al ancla
            i0 = torch.argmin(torch.abs(De))
            mask = torch.zeros_like(De, dtype=torch.bool); mask[i0] = True

        Z = Zc[mask]
        T = Te[mask]
        D = De[mask]
        Tlen = Z.shape[0]
        nW = len(self.horizons)

        # Labels multi-horizonte
        Y = torch.zeros(Tlen, nW, dtype=torch.float32)
        if target == 1.0:
            for j, h in enumerate(self.horizons):
                Y[:, j] = (D <= -float(h)).float()

        masks = torch.ones(Tlen, dtype=torch.bool)

        return {
            "id": vid,
            "embeddings": Z,          # (T, D)
            "clip_times": T,          # (T,) FIN de clip
            "deltas": D,              # (T,) FIN relativo
            "labels_tf": Y,           # (T, nW)
            "masks": masks,           # (T,)
            "target": torch.tensor(target, dtype=torch.float32),
            "meta": meta_dict,
        }
def collate_pad(batch):
    """
    Collate con padding; construye:
      - t_events (tiempo de evento real o fin de ventana)
      - durations (duraciÃƒÂ³n total del clip) -> para 'clip_end'
    OJO: usa las claves que guarda tu extractor: time_of_event, window, duration.
    """
    emb_list = [b["embeddings"] for b in batch]
    lbl_list = [b["labels_tf"] for b in batch]
    msk_list = [b["masks"] for b in batch]
    del_list = [b["deltas"] for b in batch]        # deltas al FIN del clip
    tim_list = [b["clip_times"] for b in batch]    # FIN del clip (centro+dt/2)
    tgt_list = [b["target"] for b in batch]

    t_events = []
    durations = []

    for item in batch:
        meta = item.get("meta", {}) or {}
        tgt  = float(item["target"].item())

        # duration del vÃƒÂ­deo/clip
        dur = meta.get("duration", None)
        if dur is None:
            # fallback robusto: ÃƒÂºltimo FIN observado
            dur = float(item["clip_times"].max().item())
        durations.append(float(dur))

        # tiempo de evento (positivos) o fin de ventana (negativos)
        if tgt > 0.5:
            # preferencia: time_of_event si estÃƒÂ¡
            t_event = meta.get("time_of_event", None)
            if t_event is None:
                # si no hay time_of_event: usa fin de ROI si ventana estÃƒÂ¡ndar (post=0)
                win = meta.get("window", None)
                if isinstance(win, (list, tuple)) and len(win) == 2:
                    # si la extracciÃƒÂ³n fue hist_s=5, post_s=0 Ã¢â€ â€™ end == event
                    t_event = win[1]
            if t_event is None:
                # ÃƒÂºltimo FIN observado como ÃƒÂºltimo recurso
                t_event = float(item["clip_times"].max().item())
        else:
            # negativos: evento "ficticio" = fin de ventana
            win = meta.get("window", None)
            if isinstance(win, (list, tuple)) and len(win) == 2:
                t_event = win[1]
            else:
                t_event = float(dur)

        t_events.append(float(t_event))

    embeddings = pad_sequence(emb_list, batch_first=True)
    labels_tf = pad_sequence(lbl_list, batch_first=True)
    masks     = pad_sequence(msk_list,  batch_first=True)
    deltas    = pad_sequence(del_list,  batch_first=True)
    clip_times= pad_sequence(tim_list,  batch_first=True)

    return {
        "embeddings": embeddings,   # (B,T,D)
        "labels_tf": labels_tf,     # (B,T,nW)
        "masks": masks,             # (B,T)
        "deltas": deltas,           # (B,T)  FIN - ancla
        "clip_times": clip_times,   # (B,T)  FIN
        "targets": torch.stack(tgt_list),
        "t_events": torch.tensor(t_events, dtype=torch.float32),
        "durations": torch.tensor(durations, dtype=torch.float32),
    }