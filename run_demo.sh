#!/bin/bash
# run_demo.sh
# Usage: ./run_demo.sh [video_file]

VIDEO=${1:-00023.mp4}

echo "Running Accident Anticipation Demo on $VIDEO..."
echo "Press 'q' in the video window to exit."

# Ensure PYTHONPATH includes the scripts modules
export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts/vjepa_core

python scripts/demos/demo_video.py \
  --video "$VIDEO" \
  --checkpoint checkpoints/depth_vjepa/checkpoints_vjepa_tf_rgb_frozen_hn05_j0-2_pw3/best_robust_model.pt \
  --model-type rgb \
  --head-type transformer
