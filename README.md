# V-JEPA 2 Accident Anticipation

**Official PyTorch Implementation** of the Master Thesis: *"Accident Anticipation via Video Foundation Models and Monocular Depth"*.

This repository contains the complete pipeline for training and evaluating accident anticipation models using the **V-JEPA 2** self-supervised backbone. It supports:
- **Streaming Inference**: Sliding-window processing suitable for ADAS.
- **Multi-Modal Fusion**: Integration of RGB and Monocular Depth (Depth Anything V3) via Late Fusion, Patchwise Fusion, and Mid-Fusion (FiLM).
- **Strict Pre-Event Protocol**: Evaluation preventing post-accident information leakage.

---

## 🚀 Quick Start (Demo)

We provide a script to run the pre-trained model on a raw video file.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
Run the prediction on one of the sample videos (`00023.mp4`) using the **RGB-only** robust model:

```bash
# Set PYTHONPATH for internal imports
export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts/vjepa_core

# Run Inference
python scripts/demos/demo_video.py \
  --video 00023.mp4 \
  --checkpoint checkpoints/depth_vjepa/checkpoints_vjepa_tf_rgb_frozen_hn05_j0-2_pw3/best_robust_model.pt \
  --model-type rgb \
  --no-display  # Remove this to see the video window
```
*Note: The script outputs `[PRED] t=... prob=...` log lines. Remove `--no-display` to see the visual overlay.*

---

## 📂 Repository Structure

```text
RepoUnificado/
├── checkpoints/       # Pre-trained model weights (RGB, Depth-Fusion)
├── data/              # Dataset metadata (.csv) and config
├── scripts/           # Main source code
│   ├── vjepa_core/    # Core training/eval logic (datasets, models, loops)
│   ├── depth_vjepa/   # Depth-specific modules
│   └── demos/         # Inference scripts
├── src/               # Shared libraries
└── requirements.txt   # Python dependencies
```

---

## 🛠️ Reproduction

### 1. Feature Extraction
To extract V-JEPA 2 features from your dataset videos:
```bash
python scripts/vjepa_core/extract_video_embeddings.py \
  --video_dir /path/to/raw/videos \
  --output_dir data/embeddings \
  --batch_size 16
```

### 2. Training
To train an anticipation head (e.g., Transformer Head) on extracted features:
```bash
python scripts/vjepa_core/train_nexar_vjepa_ft.py \
  --tokens_dir data/embeddings \
  --csv_train data/metadata/Nexar/train.csv \
  --head_type transformer \
  --epochs 20
```

### 3. Evaluation
To evaluate on standard benchmarks (Nexar, DAD, CCD):
```bash
python scripts/vjepa_core/eval_on_test.py \
  --checkpoint results/my_experiment/best_model.pt \
  --test_csv data/metadata/Nexar/test.csv
```

---

## 🔗 Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{GarciaHoyberg2026Thesis,
  title={Accident Anticipation via Video Foundation Models and Monocular Depth},
  author={Ander García Hoyberg},
  school={Universidad de Zaragoza},
  year={2026}
}
```
