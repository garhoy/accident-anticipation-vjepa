import numpy as np
import glob
import os

# RUTA DE TU CARPETA DE ENTRENAMIENTO (La que usaste para sacar el F1 bueno)
train_folder = "/home/ander/BADAS-Open/data/processed/Nexar_DA3_Tensors/train"

files = glob.glob(os.path.join(train_folder, "*.npz"))

if len(files) > 0:
    print(f"Inspeccionando: {files[0]}")
    data = np.load(files[0])
    depth = data['depth']
    print(f"--- DATOS DE ENTRENAMIENTO ---")
    print(f"Max: {np.max(depth)}")
    print(f"Min: {np.min(depth)}")
    print(f"Mean: {np.mean(depth)}")
else:
    print("No encontré archivos en la carpeta de train.")