# Cross-Dataset Deepfake Detection

**Rutgers MS Data Science Capstone** — Mohammad Siddikhali

Stacking ensemble of three pretrained vision models for face manipulation detection, trained and evaluated across 8 public datasets (428,743 images). Achieves **98.5% accuracy / AUC 0.9985** in-distribution, with leave-one-dataset-out (LOO) cross-validation across all 8 datasets.

---

## Results

| Model | Accuracy | AUC |
|---|---|---|
| EfficientNet-B4 | 95.8% | 0.9913 |
| Swin Transformer Tiny | 97.7% | 0.9970 |
| CLIP ViT-B/16 | 98.3% | 0.9984 |
| **Stacking Ensemble** | **98.5%** | **0.9985** |

Meta-learner coefficients (logistic regression on val split): CLIP 2.34 · Swin 1.33 · EFB4 0.39

### Datasets

| Dataset | Type |
|---|---|
| FaceForensics++ (c23) | Video deepfakes |
| Celeb-DF v2 | Celebrity deepfakes |
| 140k Real and Fake Faces | StyleGAN fakes |
| Artifact dataset | GAN artifacts |
| Real vs Fake Face | Mixed GAN |
| Deepfake and Real Images | Mixed |
| Stable Diffusion synthetic faces | Diffusion model |
| PGD adversarial fakes | Adversarial |

---

## Architecture

Three models are trained independently with model-specific hyperparameters, then combined via a logistic regression meta-learner (stacking):

- **EfficientNet-B4** — custom head on top of `tf_efficientnet_b4`; `torch.compile` enabled for ~1.3× throughput
- **Swin Transformer Tiny** — fine-tuned from ImageNet pretrained `swin_tiny_patch4_window7_224`
- **CLIP ViT-B/16** — fine-tuned from OpenAI CLIP weights via timm (`vit_base_patch16_clip_224.openai`)

Training uses SWA (Stochastic Weight Averaging) over the final N epochs via a custom `ManualSWA` implementation (PyTorch's `AveragedModel` is incompatible with `torch.compile`). Mixed precision via bfloat16, cosine LR decay with linear warmup, label smoothing (0.05), and a balanced sampler that weights each dataset and class equally.

Explainability is provided via GradCAM++ with architecture-specific hooks for all three backbones, overlaid on detected face crops.

---

## Project Structure

```
deepfake-detection/
├── app/
│   ├── main.py               # FastAPI web app (SusScan)
│   └── static/
│       └── index.html        # Frontend UI
├── configs/
│   └── default.yaml          # All hyperparameters and paths
├── scripts/
│   ├── train_ensemble.py     # Train all 3 models for one LOO holdout
│   ├── build_meta_learner.py # Fit LR stacker from existing checkpoints
│   ├── evaluate.py           # Full test-set evaluation with TTA
│   ├── predict_single.py     # Single-image inference + heatmap export
│   └── calibrate_models.py   # Fit per-model temperature scalers
├── src/
│   ├── calibration/
│   │   └── temperature_scaling.py  # Golden-section search for T
│   ├── classification/
│   │   └── verdict.py              # 7-rule 5-state verdict classifier
│   ├── data/
│   │   ├── dataset.py              # DeepfakeDataset, balanced sampler, LOO loaders
│   │   ├── face_detector.py        # MTCNN detection + square crop
│   │   ├── preprocessing.py        # CSV validation utilities
│   │   └── transforms.py           # Train / val / TTA augmentation pipelines
│   ├── evaluation/
│   │   ├── holdout.py              # LOO orchestrator + resumable sweep
│   │   └── metrics.py              # AUC, F1, detection rate, threshold search
│   ├── explainability/
│   │   ├── explain.py              # Narrative generation for predictions
│   │   ├── gradcam.py              # GradCAM++ for CNN / ViT / Swin
│   │   └── region_mapping.py       # Map heatmap activations to face regions
│   ├── inference/
│   │   ├── predict.py              # DeepfakePredictor class
│   │   └── visualize.py            # Annotated image and panel generation
│   ├── models/
│   │   ├── clip_vit.py             # CLIP ViT-B/16 wrapper
│   │   ├── efficientnet.py         # EfficientNet-B4 with custom head
│   │   ├── ensemble.py             # Load base models + meta-learner
│   │   └── swin.py                 # Swin Transformer Tiny wrapper
│   ├── training/
│   │   ├── train_clip.py           # CLIP-specific defaults
│   │   ├── train_efb4.py           # EFB4-specific defaults
│   │   ├── train_swin.py           # Swin-specific defaults
│   │   └── trainer.py              # Shared loop: SWA, loss, optimizer, eval
│   └── utils/
│       ├── config.py               # YAML config loader
│       ├── logging.py              # Filtered stderr, env printer
│       └── seed.py                 # Reproducibility + thread caps
├── tests/
│   └── test_verdict.py             # Unit tests for the 5-state verdict classifier
├── results/
│   └── confusion_matrix_140k.png   # LOO fold confusion matrix (140k StyleGAN)
├── .env.example                    # Web app environment variable template
└── notebooks/
    └── archive/                    # Original exploratory notebooks
```

---

## Installation

```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.5.1, CUDA 12.x. Developed on an RTX 5090 (24 GB VRAM).

---

## Local Setup

All paths are relative to the project root. Create the following directories and place your files there:

```
deepfake-detection/
├── checkpoints/          # model weights go here (gitignored)
│   ├── best_efb4.pt
│   ├── best_swin.pt
│   ├── best_clip.pt
│   ├── ensemble_meta.pkl
│   └── calibration/      # optional — from calibrate_models.py
│       ├── efb4_temperature.json
│       ├── swin_temperature.json
│       └── clip_temperature.json
└── data/                 # CSV and dataset image files go here (gitignored)
    └── master_v4.csv
```

Paths in `configs/default.yaml` default to `checkpoints/` and `data/` relative to the project root. You can override any path there or via the `CHECKPOINT_DIR` environment variable for the web app.

---

## Configuration

All hyperparameters and paths live in [configs/default.yaml](configs/default.yaml). The key paths to set if your files are not in the default locations:

```yaml
paths:
  csv_path: "data/master_v4.csv"
  checkpoint_dir: "checkpoints"
```

---

## Training

### Train all three models for one LOO fold

```bash
python scripts/train_ensemble.py \
    --holdout celeb_df_v2 \
    --config configs/default.yaml
```

### Train specific models only

```bash
python scripts/train_ensemble.py \
    --holdout celeb_df_v2 \
    --models swin clip
```

### Build the meta-learner from existing checkpoints

```bash
python scripts/build_meta_learner.py \
    --checkpoint-dir checkpoints \
    --csv data/master_v4.csv \
    --holdout celeb_df_v2 \
    --output checkpoints/ensemble_meta.pkl
```

### Calibrate model outputs (optional)

```bash
python scripts/calibrate_models.py
```

Outputs per-model temperature JSON files to `checkpoints/calibration/`.

---

## Evaluation

```bash
python scripts/evaluate.py \
    --holdout celeb_df_v2 \
    --checkpoint-dir checkpoints \
    --csv data/master_v4.csv \
    --output outputs/celeb_df_v2.json
```

Pass `--no-tta` to skip test-time augmentation.

---

## Inference

### Python API

```python
from src.inference.predict import DeepfakePredictor

predictor = DeepfakePredictor(checkpoint_dir="checkpoints")

# simple prediction
result = predictor.predict("face.jpg", use_tta=True)
# {'fake_prob': 0.9873, 'is_fake': True, 'model_probs': {'efb4': ..., 'swin': ..., 'clip': ...}}

# full explanation with GradCAM++ heatmaps
result = predictor.predict_with_explanation("face.jpg", use_tta=True)
# result['story'] contains per-region evidence and narrative text
```

### CLI

```bash
python scripts/predict_single.py \
    --image face.jpg \
    --checkpoint-dir checkpoints \
    --output-dir outputs/
```

---

## Web App (SusScan)

```bash
cd deepfake-detection
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`. The app detects all faces in an uploaded image using MTCNN, crops each face, runs the full ensemble + GradCAM++ pipeline, and returns a 5-state verdict (STRONG_REAL / LIKELY_REAL / UNCERTAIN / LIKELY_FAKE / STRONG_FAKE) with per-region evidence cards and model score breakdown.

To set a custom checkpoint path, use the environment variable:

```bash
CHECKPOINT_DIR=/path/to/your/checkpoints uvicorn app.main:app --port 8000
```

Or copy `.env.example` to `.env` and set `CHECKPOINT_DIR` there.

---

## Training Details

| Hyperparameter | EFB4 | Swin-T | CLIP |
|---|---|---|---|
| Total epochs | 18 | 10 | 10 |
| SWA window | 5 | 3 | 3 |
| Backbone LR | 1e-5 | 2e-5 | 1e-5 |
| Head LR | 1e-3 | 1e-3 | 5e-4 |
| Warmup epochs | 1 | 1 | 1 |
| Label smoothing | 0.05 | 0.05 | 0.05 |
| Grad clip | 1.0 | 1.0 | 1.0 |
| AMP dtype | bfloat16 | bfloat16 | bfloat16 |
| torch.compile | Yes | No | No |

All models: AdamW (β=0.9/0.999, ε=1e-8), weight decay 0.01, cosine LR with linear warmup, balanced class + dataset sampler.
