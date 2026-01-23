import torch, numpy as np
from torch.utils.data import DataLoader

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataset_embeddings import VJEPAFramewiseDataset, collate_pad

DIR_VAL = ["data_exp/vjepa2_l16s1_f10_hist5/ccd/val"]  # ajusta a tu ruta
FPS, CLIP_LEN = 10.0, 16
dt = CLIP_LEN / FPS

val_ds = VJEPAFramewiseDataset(DIR_VAL, window_s=5.0, anticipation_windows=[1,2,3,5])
val_ld = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_pad)

t_events, t_first_center, t_first_start, t_first_end = [], [], [], []

with torch.no_grad():
    for b in val_ld:
        mask = b["masks"]              # (B,T)
        tfin = b["clip_times"]         # (B,T) FIN (Te)
        tcen = tfin - dt/2             # CENTER
        tsta = tfin - dt               # START
        tend = tfin                    # END

        # primer tiempo válido (por vídeo) según decisión
        first_center = torch.where(mask, tcen, torch.full_like(tcen, 1e9)).min(dim=1).values
        first_start  = torch.where(mask, tsta, torch.full_like(tsta, 1e9)).min(dim=1).values
        first_end    = torch.where(mask, tend, torch.full_like(tend, 1e9)).min(dim=1).values

        t_events.extend(b["t_events"].tolist())
        t_first_center.extend(first_center.tolist())
        t_first_start.extend(first_start.tolist())
        t_first_end.extend(first_end.tolist())

t_events = np.array(t_events); t_first_center = np.array(t_first_center)
t_first_start = np.array(t_first_start); t_first_end = np.array(t_first_end)

print(f"Evento medio: {t_events.mean():.3f} s")
print(f"cap_center = {np.mean(t_events - t_first_center):.3f} s")
print(f"cap_start  = {np.mean(t_events - t_first_start):.3f} s")
print(f"cap_end    = {np.mean(t_events - t_first_end):.3f} s")
