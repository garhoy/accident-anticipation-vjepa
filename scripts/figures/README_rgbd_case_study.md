# RGBD Case Study Figure

Generates a single PNG (and optional PDF) with 5 RGB frames, 5 depth frames, and
the RGB-only vs RGB+Depth risk curves for the same clip.

Example (depth video):
```bash
python scripts/figures/plot_rgbd_case_study.py \
  --rgb-video 00037.mp4 \
  --depth-video 00037_depth.mp4 \
  --curve-rgb results/curve_rgb_00037.csv \
  --curve-rgbd results/curve_rgbd_00037.csv \
  --times 15.5 16.0 16.5 17.0 17.5 \
  --t-event 20.533 \
  --theta 0.5 \
  --out Imagenes/Fig_RGB_Depth_Qualitative.png \
  --pdf
```

Example (depth frames directory):
```bash
python scripts/figures/plot_rgbd_case_study.py \
  --rgb-video 00037.mp4 \
  --depth-frames-dir data/depth_frames/00037 \
  --depth-fps 30 \
  --curve-rgb results/curve_rgb_00037.csv \
  --curve-rgbd results/curve_rgbd_00037.csv \
  --times 15.5 16.0 16.5 17.0 17.5 \
  --t-event none \
  --theta 0.5 \
  --out Imagenes/Fig_RGB_Depth_Qualitative.png \
  --export-debug Imagenes/debug_case_study
```

Before (legacy layout, full x-axis):
```bash
python scripts/figures/plot_rgbd_case_study.py \
  --rgb-video 00037.mp4 \
  --depth-video Imagenes/00037_depth_full.mp4 \
  --curve-rgb results/demo_stream_rgb_00037.csv \
  --curve-rgbd results/demo_stream_rgbd_midfilm_00037.csv \
  --times 15.5 16.0 16.5 17.0 17.5 \
  --t-event 20.533 \
  --theta 0.5 \
  --out Imagenes/Fig_RGB_Depth_Qualitative.png
```

After (paper mode, zoomed x-axis):
```bash
python scripts/figures/plot_rgbd_case_study.py \
  --rgb-video 00037.mp4 \
  --depth-video Imagenes/00037_depth_full.mp4 \
  --curve-rgb results/demo_stream_rgb_00037.csv \
  --curve-rgbd results/demo_stream_rgbd_midfilm_00037.csv \
  --times 15.5 16.0 16.5 17.0 17.5 \
  --t-event 20.533 \
  --theta 0.5 \
  --paper \
  --out Imagenes/Fig_RGB_Depth_Qualitative.png
```
