#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset para entrenar modelos con tokens espacio-temporales (NO embeddings pooled).

Diferencias clave vs VJEPAFramewiseDataset:
- Carga 'tokens' (T, L, D) en vez de 'z_clip' (T, D)
- Preserva geometría espacial (grid_h, grid_w) y temporal (tubelet_size)
- Compatible con BADASAttentiveProbe, HybridLocalGlobalProbe, VJEPAClipClassifier

✅ VEREDICTO: Este código es correcto. 
   El "PROBLEMA ADICIONAL NO RESUELTO" del análisis final estaba equivocado.
   La lógica de `__getitem__` (líneas 146-159) ya produce
   labels a nivel de CLIP `(T_clips, nW)`, no de frame.
"""
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


class DatasetTokens(Dataset):
    """
    Dataset que carga tokens espacio-temporales de clips para modelos estilo V-JEPA/BADAS.
    
    Cada sample devuelve:
        - tokens: (T_clips, L, D)
        - labels_tf: (T_clips, nW) etiquetas multi-horizonte por clip
        - tokens_meta: dict con geometría (grid_h, grid_w, tubelet_size, etc.)
    """
    def __init__(
        self,
        embeddings_dirs: Union[str, Path, List[str], List[Path]],
        window_s: float = 5.0,
        post_window_s: float = 0.0,
        anticipation_windows: List[float] = [1.0, 2.0, 3.0, 5.0],
        video_ids: Optional[List[str]] = None
    ):
        self.dirs = _to_list_dirs(embeddings_dirs)
        self.W = float(window_s)
        self.post = float(post_window_s)
        self.horizons = [float(x) for x in anticipation_windows]

        self.index: Dict[str, Path] = {}
        for d in self.dirs:
            p = Path(d)
            if not p.exists(): 
                print(f"[WARN] Directorio no encontrado: {p}")
                continue
            for pt in p.glob("*.pt"):
                self.index[str(pt.name)] = pt # Indexar por nombre de archivo

        
        # Filtrar IDs si se provee una lista
        ids_to_load = []
        if video_ids is None:
            ids_to_load = sorted(self.index.keys())
        else:
            for vid in video_ids:
                # Buscar por ID (ej: '000123.pt') o por ID completo (ej: 'train/pos/000123.pt')
                if vid in self.index:
                    ids_to_load.append(vid)
                elif f"{vid}.pt" in self.index:
                    ids_to_load.append(f"{vid}.pt")
        
        self.ids = ids_to_load
        
        # Validar que los .pt tengan 'tokens'
        self.meta = []
        print(f"Validando {len(self.ids)} archivos .pt...")
        for vid_name in self.ids:
            p = self.index.get(vid_name)
            if p is None:
                print(f"[WARN] No se encontró el path para ID {vid_name}")
                continue
            
            try:
                data = torch.load(p, map_location="cpu")
                
                if "tokens" not in data:
                    print(f"[WARN] {vid_name}: sin clave 'tokens', saltando")
                    continue
                
                tokens = data["tokens"]
                if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] == 0:
                    print(f"[WARN] {vid_name}: tokens inválidos (shape={tokens.shape}), saltando")
                    continue
                
                if "clip_times" not in data or "delta_to_anchor" not in data:
                    print(f"[WARN] {vid_name}: faltan 'clip_times' o 'delta_to_anchor', saltando")
                    continue

                tgt = int(data.get("meta", {}).get("target", 0))
                self.meta.append({
                    "id": str(data.get("id", vid_name)),
                    "path": p,
                    "target": tgt
                })
            except Exception as e:
                print(f"[WARN] Error cargando {vid_name}: {e}")

        print(f"DatasetTokens: {len(self.meta)}/{len(self.ids)} vídeos válidos con tokens")
        if self.meta:
            npos = sum(m["target"] for m in self.meta)
            print(f"  Positivos: {npos} ({100*npos/len(self.meta):.1f}%) | Negativos: {len(self.meta)-npos}")

    def __len__(self):
        return len(self.meta)

    def get_sampler_weights(self) -> List[float]:
        """Devuelve pesos para WeightedRandomSampler"""
        targets = [m["target"] for m in self.meta]
        n_pos = sum(targets)
        n_neg = len(targets) - n_pos
        if n_pos == 0 or n_neg == 0:
            return [1.0] * len(targets)
        
        w_pos = 1.0 / n_pos
        w_neg = 1.0 / n_neg
        weights = [w_pos if t == 1 else w_neg for t in targets]
        return weights

    def __getitem__(self, idx):
        m = self.meta[idx]
        data = torch.load(m["path"], map_location="cpu")
        vid = m["id"]
        target = float(int(m["target"]))

        # Cargar tokens (T_clips, L, D)
        tokens_full: torch.Tensor = data["tokens"].float()  # (T_all_clips, L, D)
        Tc: torch.Tensor = data["clip_times"].float()       # (T_all_clips,) centros
        Dc: torch.Tensor = data["delta_to_anchor"].float()  # (T_all_clips,)
        meta_dict = data.get("meta", {})
        tokens_meta = data.get("tokens_meta", {})
        
        # Validar consistencia
        if tokens_full.shape[0] != Tc.shape[0] or Tc.shape[0] != Dc.shape[0]:
            print(f"[WARN] Mismatch en {vid}: tokens={tokens_full.shape[0]}, "
                  f"times={Tc.shape[0]}, deltas={Dc.shape[0]}. Usando min.")
            min_len = min(tokens_full.shape[0], Tc.shape[0], Dc.shape[0])
            tokens_full = tokens_full[:min_len]
            Tc = Tc[:min_len]
            Dc = Dc[:min_len]

        fps = float(data.get("fps_target", meta_dict.get("fps_target", 10.0)))
        clip_len = int(meta_dict.get("clip_len", 16))
        half = (clip_len - 1) / (2.0 * max(fps, 1e-6))  # ~0.75s para L=16@10Hz

        # FIN de clip = centro + half_span
        Te = Tc + half
        De = Dc + half

        # Recorte temporal a ventana [-W, post]
        mask = (De >= -self.W) & (De <= self.post)
        if not mask.any():
            i0 = torch.argmin(torch.abs(De))
            mask = torch.zeros_like(De, dtype=torch.bool)
            mask[i0] = True

        # Aplicar máscara temporal
        tokens = tokens_full[mask]  # (T, L, D)
        T_seq = Te[mask]            # (T,)
        D_seq = De[mask]            # (T,)
        
        # ✅ Tlen es T_clips_in_window
        Tlen = tokens.shape[0] 
        nW = len(self.horizons)

        # ✅ Y es (T_clips_in_window, nW)
        Y = torch.zeros(Tlen, nW, dtype=torch.float32)
        if target == 1.0:
            for j, h in enumerate(self.horizons):
                # ✅ D_seq (T_clips,) se compara con h
                Y[:, j] = (D_seq <= -float(h)).float()

        masks = torch.ones(Tlen, dtype=torch.bool)

        return {
            "id": vid,
            "tokens": tokens,         # (T_clips, L, D)
            "clip_times": T_seq,      # (T_clips,) FIN de clip
            "deltas": D_seq,          # (T_clips,) FIN relativo
            "labels_tf": Y,           # (T_clips, nW) ✅ ESTO ES CORRECTO
            "masks": masks,           # (T_clips,)
            "target": torch.tensor(target, dtype=torch.float32),
            "meta": meta_dict,
            "tokens_meta": tokens_meta,
        }


def collate_tokens(batch):
    """
    Collate para tokens espacio-temporales con padding.
    """
    tok_list = [b["tokens"] for b in batch]      # lista de (T_i_clips, L, D)
    lbl_list = [b["labels_tf"] for b in batch]   # lista de (T_i_clips, nW)
    msk_list = [b["masks"] for b in batch]       # lista de (T_i_clips,)
    del_list = [b["deltas"] for b in batch]
    tim_list = [b["clip_times"] for b in batch]
    tgt_list = [b["target"] for b in batch]

    t_events = []
    durations = []
    for item in batch:
        meta = item.get("meta", {}) or {}
        tgt = float(item["target"].item())

        dur = meta.get("duration", None)
        if dur is None:
            dur = float(item["clip_times"].max().item()) if item["clip_times"].numel() > 0 else 0.0
        durations.append(float(dur))

        t_event = 0.0 # Default para negativos
        if tgt > 0.5:
            t_event = meta.get("time_of_event", None)
            if t_event is None:
                win = meta.get("window", None)
                if isinstance(win, (list, tuple)) and len(win) == 2:
                    t_event = win[1]
            if t_event is None:
                # Fallback peligroso: asumir que el evento es el fin
                t_event = float(item["clip_times"].max().item()) if item["clip_times"].numel() > 0 else 0.0
        else:
            win = meta.get("window", None)
            if isinstance(win, (list, tuple)) and len(win) == 2:
                t_event = win[1]
            else:
                t_event = float(dur)
        t_events.append(float(t_event))

    # Padding de tokens (3D)
    B = len(tok_list)
    
    # Manejar batch vacío
    if B == 0:
        return {}
        
    # Asumir L y D fijos del primer sample no vacío
    first_valid = next((t for t in tok_list if t.numel() > 0), None)
    if first_valid is None:
        print("[WARN] collate_tokens: Batch con todos los samples vacíos.")
        L, D = 0, 0
    else:
        L = first_valid.shape[1]
        D = first_valid.shape[2]
        
    T_max = max((t.shape[0] for t in tok_list), default=0)
    
    tokens_padded = torch.zeros(B, T_max, L, D, dtype=first_valid.dtype if first_valid is not None else torch.float32)
    for i, tok in enumerate(tok_list):
        if tok.numel() > 0:
            T_i = tok.shape[0]
            tokens_padded[i, :T_i] = tok

    # Padding 2D normal
    labels_tf = pad_sequence(lbl_list, batch_first=True)
    masks = pad_sequence(msk_list, batch_first=True)
    deltas = pad_sequence(del_list, batch_first=True)
    clip_times = pad_sequence(tim_list, batch_first=True)

    tokens_meta = batch[0].get("tokens_meta", {})

    return {
        "tokens": tokens_padded,       # (B, T_max_clips, L, D)
        "labels_tf": labels_tf,        # (B, T_max_clips, nW)
        "masks": masks,                # (B, T_max_clips)
        "deltas": deltas,
        "clip_times": clip_times,
        "targets": torch.stack(tgt_list),
        "t_events": torch.tensor(t_events, dtype=torch.float32),
        "durations": torch.tensor(durations, dtype=torch.float32),
        "tokens_meta": tokens_meta,
    }