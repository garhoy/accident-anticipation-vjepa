#!/usr/bin/env python3
# test_causal.py
import torch
import sys
from pathlib import Path

# Importar tu modelo
sys.path.insert(0, str(Path(__file__).parent))
from models import TransformerPerFrame

print("="*60)
print("TEST DE CAUSALIDAD - TransformerPerFrame")
print("="*60)

# Crear modelo
model = TransformerPerFrame(
    embed_dim=1024, 
    n_windows=4,
    d_model=256,
    n_heads=4,
    num_layers=2
)
model.eval()

# Datos de prueba
B, T, D = 2, 10, 1024
x = torch.randn(B, T, D)
mask = torch.ones(B, T, dtype=torch.bool)

# Test: ¿Frame 3 cambia si modifico frame 7?
print("\n1. Predicción original...")
with torch.no_grad():
    out_original = model(x.clone(), mask)
    pred_frame3_original = out_original[:, 3, :].clone()

print("2. Modificando frame 7 (FUTURO respecto a frame 3)...")
x_modified = x.clone()
x_modified[:, 7, :] = 999.0  # Cambio drástico en el futuro

with torch.no_grad():
    out_modified = model(x_modified, mask)
    pred_frame3_modified = out_modified[:, 3, :].clone()

# Comparar
diff = (pred_frame3_original - pred_frame3_modified).abs().max().item()
print(f"3. Diferencia máxima en frame 3: {diff:.6f}")

print("\n" + "="*60)
if diff > 1e-5:
    print("❌ RESULTADO: TRANSFORMER NO ES CAUSAL")
    print("   El frame 3 cambió cuando modificaste el frame 7 (futuro)")
    print("   Tu modelo está VIENDO EL FUTURO")
    print("   → El 0.88 AP es INVÁLIDO para anticipación")
else:
    print("✅ RESULTADO: TRANSFORMER ES CAUSAL")
    print("   El frame 3 NO cambió cuando modificaste el frame 7")
    print("   Tu modelo NO ve el futuro")
    print("   → El 0.88 AP es VÁLIDO")
print("="*60)