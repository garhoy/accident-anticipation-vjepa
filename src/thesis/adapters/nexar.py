#!/usr/bin/env python3
"""
NEXAR DATASET ADAPTER

Este adapter hace 3 cosas principales:
1. Lee la configuración de nexar.yaml
2. Carga videos desde disco
3. Muestrea frames según el protocolo temporal definido

FLUJO:
    config = load_config("nexar.yaml")
    dataset = NexarDataset(config, split="train")
    
    for video, label, video_id in dataset:
        # video: torch.Tensor [T, C, H, W]
        # T = frames (ej: 100)
        # C = canales (3 para RGB)
        # H, W = altura, ancho (ej: 224, 224)
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

import torch
import yaml
import cv2
import numpy as np

# ============================================================================
# CARGAR CONFIGURACIÓN
# ============================================================================

def load_config(yaml_path: str) -> Dict:
    """
    Carga el archivo YAML de configuración y expande los paths.
    
    Args:
        yaml_path: Ruta al nexar.yaml
    
    Returns:
        Diccionario con toda la configuración
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expandir ~ en los paths
    for key in ['videos_root', 'metadata_root', 'embeddings_root']:
        if key in config.get('paths', {}):
            config['paths'][key] = str(Path(config['paths'][key]).expanduser())
    
    return config

# ============================================================================
# MUESTREO TEMPORAL DE VIDEO
# ============================================================================

def sample_video_frames(
    video_path: Path,
    target_fps: int,
    history_seconds: float,
    strategy: Literal["random", "tail", "full"],
    original_fps: float = 30.0
) -> np.ndarray:
    """
    Carga un video y muestrea frames según la estrategia.
    
    PARÁMETROS EXPLICADOS:
    
    video_path: 
        - Ruta al archivo .mp4
        
    target_fps: 
        - FPS al que quieres muestrear (ej: 20)
        - Si el video original es 30 FPS y pones target=20:
          → Tomarás 1 frame cada 1.5 frames originales
        
    history_seconds: 
        - Cuántos segundos extraer (ej: 5.0)
        - Con target_fps=20 y history=5 → 100 frames
        
    strategy:
        - "tail": Toma los últimos N segundos
          Ejemplo: Video de 40s, history=5s → frames del segundo 35-40
        
        - "random": Toma una ventana aleatoria de N segundos
          Ejemplo: Video de 40s, history=5s → puede ser 10-15s, o 20-25s, etc.
        
        - "full": Toma TODO el video (ignora history_seconds)
          NO recomendado, genera muchos frames
    
    Returns:
        np.ndarray [T, H, W, C] donde:
        - T = número de frames (ej: 100)
        - H, W = altura/ancho del video original (1280, 720)
        - C = 3 (BGR, formato de OpenCV)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")
    
    # Propiedades del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = original_fps
    
    # Cuántos frames necesitamos al final
    n_frames_needed = int(history_seconds * target_fps)
    
    # ========== ESTRATEGIA DE MUESTREO ==========
    
    if strategy == "full":
        # TODO el video
        start_frame = 0
        end_frame = total_frames
        
    elif strategy == "tail":
        # Últimos N segundos
        # Ejemplo: Video 40s (1200 frames @ 30fps), history=5s
        # → end_frame = 1200
        # → start_frame = 1200 - (5*30) = 1050
        # → Frames 1050-1200 (últimos 5 segundos)
        end_frame = total_frames
        start_frame = max(0, end_frame - int(history_seconds * video_fps))
        
    elif strategy == "random":
        # Ventana aleatoria
        window_frames = int(history_seconds * video_fps)
        if window_frames >= total_frames:
            start_frame = 0
            end_frame = total_frames
        else:
            # Elegir inicio aleatorio
            start_frame = random.randint(0, total_frames - window_frames)
            end_frame = start_frame + window_frames
    
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")
    
    # ========== CALCULAR STRIDE PARA DOWNSAMPLING ==========
    
    # Si el video es 30 FPS y quieres 20 FPS:
    # frame_stride = 30 / 20 = 1.5
    # → Tomarás frames 0, 1, 3, 4, 6, 7, 9...
    frame_stride = max(1, int(video_fps / target_fps))
    
    # ========== EXTRAER FRAMES ==========
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Muestrear cada frame_stride frames
        if (current_frame - start_frame) % frame_stride == 0:
            frames.append(frame)
        
        current_frame += 1
        
        # Si ya tenemos suficientes, parar
        if len(frames) >= n_frames_needed:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError(f"No se extrajo ningún frame de {video_path}")
    
    # ========== PADDING SI FALTAN FRAMES ==========
    
    # Si el video era muy corto y no llegamos a n_frames_needed,
    # repetir el último frame
    while len(frames) < n_frames_needed:
        frames.append(frames[-1].copy())
    
    # Truncar si tenemos de más
    frames = frames[:n_frames_needed]
    
    return np.stack(frames, axis=0)  # [T, H, W, C]

# ============================================================================
# CONVERSIÓN A TENSOR PYTORCH
# ============================================================================

def frames_to_tensor(
    frames: np.ndarray, 
    resize: Optional[int] = None
) -> torch.Tensor:
    """
    Convierte frames numpy → tensor PyTorch.
    
    PARÁMETROS:
    
    frames: 
        - np.ndarray [T, H, W, C] en formato BGR (OpenCV)
        
    resize: 
        - Si se especifica (ej: 224), redimensiona TODOS los frames a (224, 224)
        - Si es None, mantiene resolución original
    
    Returns:
        torch.Tensor [T, C, H, W] donde:
        - Canales en formato RGB (no BGR)
        - Valores normalizados a [0, 1]
        - Tipo: float32
    """
    # BGR → RGB
    frames_rgb = frames[:, :, :, ::-1].copy()
    
    # Redimensionar si es necesario
    if resize is not None:
        frames_resized = []
        for frame in frames_rgb:
            frame_small = cv2.resize(frame, (resize, resize))
            frames_resized.append(frame_small)
        frames_rgb = np.stack(frames_resized, axis=0)
    
    # Numpy [T, H, W, C] → Tensor [T, C, H, W]
    tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float()
    
    # Normalizar a [0, 1]
    tensor = tensor / 255.0
    
    return tensor

# ============================================================================
# DATASET CLASS
# ============================================================================

class NexarDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset para Nexar.
    
    USO:
        dataset = NexarDataset(config, split="train", resize=224)
        video, label, video_id = dataset[0]
    
    DEVUELVE:
        video: torch.Tensor [T, C, H, W]
               - T = número de frames (ej: 100)
               - C = 3 (RGB)
               - H, W = resize (ej: 224, 224)
               - Valores en [0, 1]
        
        label: int
               - 0 = negativo (sin colisión)
               - 1 = positivo (con colisión)
        
        video_id: str
                  - Identificador del video (ej: "positive/00003")
    """
    
    def __init__(
        self,
        config: Dict,
        split: Literal["train", "val"],
        resize: Optional[int] = None
    ):
        """
        PARÁMETROS:
        
        config: 
            - Diccionario cargado con load_config()
            
        split: 
            - "train" o "val"
            - Determina qué CSV cargar (train.csv o val.csv)
            
        resize: 
            - Resolución a la que redimensionar frames (ej: 224)
            - Si None, mantiene resolución original (1280x720)
            - RECOMENDADO: 224 (para V-JEPA) o 256 (para BADAS)
        """
        self.config = config
        self.split = split
        self.resize = resize
        
        # Paths
        self.videos_root = Path(config['paths']['videos_root'])
        metadata_root = Path(config['paths']['metadata_root'])
        
        # Cargar metadata CSV
        csv_filename = config['splits'][split]['csv']
        csv_path = metadata_root / csv_filename
        self.samples = self._load_csv(csv_path)
        
        # Parámetros de muestreo
        self.fps = config['sampling']['fps']
        self.history_seconds = config['sampling']['history_seconds']
        self.original_fps = config['video_specs']['original_fps']
        
        # Estrategias
        self.positive_strategy = config['sampling']['positive_strategy']
        self.negative_strategy = config['sampling']['negative_strategy']
        
        print(f"✓ NexarDataset [{split}]: {len(self.samples)} samples")
    
    def _load_csv(self, csv_path: Path) -> List[Dict]:
        """Carga el CSV con los IDs y labels."""
        samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    'id': row['id'],        # ej: "positive/00003"
                    'label': int(row['label'])  # 0 o 1
                })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Carga un video y lo procesa."""
        sample = self.samples[idx]
        video_id = sample['id']  # "positive/00003"
        label = sample['label']  # 0 o 1
        
        # Elegir estrategia según el label
        strategy = self.positive_strategy if label == 1 else self.negative_strategy
        
        # Path al video
        video_path = self.videos_root / f"{video_id}.mp4"
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        # Cargar frames
        frames = sample_video_frames(
            video_path,
            target_fps=self.fps,
            history_seconds=self.history_seconds,
            strategy=strategy,
            original_fps=self.original_fps
        )
        
        # Convertir a tensor
        video_tensor = frames_to_tensor(frames, resize=self.resize)
        
        return video_tensor, label, video_id

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='nexar.yaml')
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--n-samples', type=int, default=3)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"TESTING NEXAR ADAPTER [{args.split}]")
    print("="*60 + "\n")
    
    config = load_config(args.config)
    dataset = NexarDataset(config, split=args.split, resize=224)
    
    print(f"\nCargando {args.n_samples} samples...\n")
    
    for i in range(min(args.n_samples, len(dataset))):
        video, label, video_id = dataset[i]
        print(f"[{i}] {video_id}")
        print(f"    Shape: {video.shape}")
        print(f"    Label: {label}")
        print(f"    Range: [{video.min():.3f}, {video.max():.3f}]")
        print()