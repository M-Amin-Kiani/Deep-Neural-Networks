# EfficientAD++: Unique-Map LOCO Evaluation & Post-processing Improvements

> **Status:** Research / Reproducibility project  
> **Base implementation:** https://github.com/nelson1425/EfficientAD (upstream reference)  
> **Focus:** Fixing LOCO pixel-level evaluation issues (NaN) + adding robust, memory-safe metrics + optional post-processing.

---

## 1) Overview

This repository is a **research-oriented extension** of the original EfficientAD implementation.  
The goal is to make EfficientAD **more reproducible and evaluation-correct** on **MVTec AD** and **MVTec LOCO** inside constrained environments (e.g., Google Colab), where users commonly face:

- broken/expired dataset download links
- RAM crashes during pixel-metric computation
- **NaN pixel AUROC on MVTec LOCO** due to **anomaly-map filename collisions**
- missing / incompatible official evaluation scripts across forks or notebooks

### What we improved vs. the base repository

1. **LOCO “Unique Anomaly Map” Saving (Critical Fix)**
   - MVTec LOCO contains repeated filenames (e.g., `000.png`) across subfolders like:
     ```
     test/logical_anomalies/068/000.png
     test/logical_anomalies/014/000.png
     ```
   - If anomaly maps are saved only by filename (`000.tiff`), maps overwrite each other.
   - This causes evaluation to effectively see **no positive pixels**, producing:
     - `pos_pixels = 0`
     - `pixel_auroc = NaN`
   - **Fix:** Save anomaly maps with a unique name derived from the relative path:
     ```
     logical_anomalies__068__000.tiff
     logical_anomalies__014__000.tiff
     ```

2. **Memory-safe pixel metrics**
   - Pixel AUROC and AU-PRO can consume large RAM.
   - We provide **streaming / sampling** metrics to avoid crashes while remaining informative.

3. **Optional post-processing (research idea)**
   - We provide an optional post-processing pipeline (e.g., `dcc_post`) to explore improvements beyond baseline anomaly maps.

---

## 2) What is EfficientAD?

EfficientAD is a fast anomaly detection method designed for industrial visual inspection.  
It typically reports metrics such as:

- **Image-level AUROC**
- **Pixel-level AUROC**
- **PRO / AU-PRO** (Region-based overlap measure)

> This repo does not rewrite EfficientAD from scratch.  
> We **build on top of the original implementation** and focus on **evaluation correctness + reproducible pipelines**.

---

## 3) Key Contributions (This Project)

### (A) Correct Pixel Evaluation on MVTec LOCO (No more NaN)

**Problem:** LOCO test images share repeated filenames → anomaly maps overwrite → pixel AUROC becomes NaN.  
**Solution:** Re-generate anomaly maps with unique filenames based on relative path.

✅ Expected result:

- `pos_pixels > 0`
- pixel-level metrics become computable and reportable

---

### (B) Reliable, Low-RAM Pixel Metrics

We provide:

- **streaming pixel AUROC** using sampled pixels
- **AU-PRO@0.3** computed on a bounded subset (configurable)

This enables stable evaluation inside Colab without runtime disconnections.

---

### (C) Outputs: TensorBoard + Weights + One-page report

This project produces:

- TensorBoard logs for **training & testing**
- final weights for each run:
  - `teacher_final.pth`, `student_final.pth`, `autoencoder_final.pth`
- CSV summaries:
  - `metrics_summary.csv`
  - pixel metrics CSV outputs (sampled / reliable)
- optionally a 1-page PDF summary

---

## 4) Repository Structure (Typical)

```
.
├── EfficientAD/                           # upstream code (or fork)
│   ├── efficientad_tb.py                  # training + inference + TB logging
│   ├── efficientad_loco_infer_unique.py   # NEW: unique LOCO map saving
│   ├── efficientad_dcc_post.py            # optional post-processing experiments
│   └── ...
├── outputs/
│   ├── efficientad_runs/
│   │   ├── trainings/
│   │   ├── anomaly_maps/
│   │   ├── metrics_summary.csv
│   │   └── pixel_metrics_*.csv
│   └── ...
└── README.md
```

---

## 5) Requirements

### Recommended environment (Google Colab)

- Python 3.10+ (Colab often uses 3.10/3.11/3.12)
- CUDA runtime recommended (T4/V100/A100)
- Packages installed by the original EfficientAD repo + additional:
  - `tifffile`, `scikit-image`, `tensorboard`, `pandas`

---

## 6) Quickstart (Colab-friendly)

### Step 1 — Clone EfficientAD (Base)

```bash
%cd /content
!git clone --depth 1 https://github.com/nelson1425/EfficientAD.git
%cd /content/EfficientAD
```

### Step 2 — Download datasets

Due to frequent link failures / large archives, we support:

- full download (if possible)
- fallback (single category) when full download fails

You should end with:

- `/content/datasets/mvtec_anomaly_detection/<category>/...`
- `/content/datasets/mvtec_loco_anomaly_detection/<category>/...`

> Note: MVTec dataset distribution links may change or throttle.  
> If full downloads fail in Colab, use category-level fallback.

---

## 7) Run EfficientAD (Training + Inference)

Outputs are produced under:
`/content/outputs/efficientad_runs`

Example:

```bash
python efficientad_tb.py \
  --dataset mvtec_ad \
  --subdataset bottle \
  --mvtec_ad_path /content/datasets/mvtec_anomaly_detection \
  --output_dir /content/outputs/efficientad_runs \
  --model_size medium \
  --weights /content/EfficientAD/models/teacher_medium.pth \
  --train_steps 7000 \
  --imagenet_train_path /content/datasets/imagenette2-160/train
```

---

## 8) CRITICAL: Fix LOCO Pixel Metrics (Unique Map Saving)

### Why this is needed

MVTec LOCO uses repeated filenames (e.g., `000.png`), so saving anomaly maps as `000.tiff`
causes overwriting and invalid pixel evaluation.

### Fix: Re-run LOCO inference with unique map names

We generate anomaly maps named by relative path, e.g.:
`logical_anomalies__068__000.tiff`

Run:

```bash
python efficientad_loco_infer_unique.py \
  --subdataset breakfast_box \
  --mvtec_loco_path /content/datasets/mvtec_loco_anomaly_detection \
  --train_dir /content/outputs/efficientad_runs/trainings/mvtec_loco/breakfast_box \
  --out_maps_dir /content/outputs/efficientad_runs/anomaly_maps/mvtec_loco \
  --model_size medium \
  --device cuda
```

✅ After this, LOCO pixel AUROC should no longer become NaN.

---

## 9) Compute Metrics

### Image-level AUROC

Saved automatically by the training script (and typically written into `final_metrics.json`).

### Pixel-level metrics (Memory-safe)

We provide sampling/streaming scripts to compute:

- pixel AUROC
- AU-PRO@0.3
  without consuming excessive RAM.

Outputs:

- `pixel_metrics_custom_sampled.csv`
- `pixel_metrics_reliable.csv` (recommended)

---

## 10) TensorBoard Logging

Training and test metrics are logged automatically:

```bash
%load_ext tensorboard
%tensorboard --logdir /content/outputs/efficientad_runs/trainings
```

You can inspect:

- loss curves
- final image AUROC
- pixel metrics (if logged)

---

## 11) Model Weights Report

For each dataset/subdataset:

- `teacher_final.pth`
- `student_final.pth`
- `autoencoder_final.pth`

A summary CSV can be produced:

- `weights_report.csv` (paths + sizes)

---

## 12) One-page PDF Report (Optional)

We support producing a simple A4 PDF summary including:

- executed categories
- mean image AUROC
- pixel metrics
- comparison to reported paper numbers (when applicable)

> Important: If you run only a subset of categories (e.g., fallback downloads),
> your mean will not match paper-wide means. Mention this clearly in reports.

---

## 13) Citation & Acknowledgements

### Base repository

This work is built on top of EfficientAD implementation:

- https://github.com/nelson1425/EfficientAD

Please cite the original EfficientAD paper and codebase when using this repository.

### Our contribution statement

We extend the base implementation with:

- LOCO-safe anomaly map naming (prevents overwrite)
- reproducible low-RAM pixel metrics
- optional post-processing experiments

These improvements aim to make EfficientAD experiments:

- **more stable**
- **more reproducible**
- **evaluation-correct** (especially on LOCO)

---

## 14) Notes on Reproducibility

- Ensure consistent:
  - `train_steps`
  - model size (`small` / `medium`)
  - penalty setting (`imagenet_train_path`)
- LOCO pixel evaluation requires the unique-map fix
- Full dataset replication requires downloading all categories

---

## 15) License

This repository follows the licensing of the upstream EfficientAD repository.
Please check the upstream `LICENSE` file and comply with dataset usage terms.

---

## Contact / Maintainers

- Project author: _(Mohammad Amin Kiani)_
- For issues: open a GitHub issue with:
  - logs
  - command used
  - dataset structure listing
  - sample paths of anomaly maps and ground truth
