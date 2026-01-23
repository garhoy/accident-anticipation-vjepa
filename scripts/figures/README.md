# RGB-only Figures (Nexar)

Mini-package of reproducible scripts to generate RGB-only thesis figures.

Dependencies:
- numpy
- matplotlib
- pandas
- opencv-python
- (optional) pyyaml for `--config`

All scripts accept CSV/JSON/NPY curves with:
- CSV: columns `time_s`, `prob` (aliases accepted)
- JSON: `{"time_s":[...], "prob":[...]}`
- NPY: Nx2 array `[time, prob]`

## FIG1: Qualitative risk curves

Two-subplot figure (1x2) with t_event, theta, t_alarm, and TTA.

```bash
python scripts/figures/plot_risk_curves.py \
  --input-a results/curve_early.csv \
  --input-b results/curve_miss.csv \
  --t-event 20.53 \
  --theta 0.5 \
  --modeltag transformer \
  --out Imagenes/rgb_risk_curves_transformer.png
```

## FIG2: Adaptation capacity

AP/AUC vs #unfrozen blocks. Optional mTTA on secondary axis.

CSV expected columns (case-insensitive):
- head, unfreeze, ap, auc, mtta (optional)

```bash
python scripts/figures/plot_adaptation_capacity.py \
  --metrics-csv results/metrics_rgb.csv \
  --layout subplots \
  --mtta-mode secondary-line \
  --out Imagenes/adaptation_capacity_rgb.png
```

## FIG3: Case study (frames + prob)

Extract 5 frames from a video and plot them above the curve with markers.

```bash
python scripts/figures/plot_case_study_frames_prob.py \
  --video 00037.mp4 \
  --curve results/demo_stream_rgb_00037.csv \
  --t-event 20.533 \
  --times 15.0 16.0 17.0 18.0 19.0 \
  --theta 0.5 \
  --out Imagenes/casestudy_frames_prob_00037.png
```

## Optional YAML config

Each script accepts `--config path.yaml`. CLI args override YAML values.

Example (plot_risk_curves.yaml):
```yaml
input_a: results/curve_early.csv
input_b: results/curve_miss.csv
t_event: 20.53
theta: 0.5
modeltag: transformer
out: Imagenes/rgb_risk_curves_transformer.png
```
