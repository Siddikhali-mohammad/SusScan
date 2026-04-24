import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CSV_PATH       = PROJECT_ROOT / "data" / "master_v4.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CALIB_DIR      = CHECKPOINT_DIR / "calibration"

MAX_PER_CLASS = 1000  # up to 1000 real + 1000 fake


def _check_prerequisites():
    if not CSV_PATH.exists():
        print(f"  WARNING: CSV not found at {CSV_PATH}. Skipping calibration.")
        return False
    if not CHECKPOINT_DIR.exists():
        print(f"  WARNING: Checkpoint dir not found at {CHECKPOINT_DIR}. Skipping calibration.")
        return False
    return True


def _load_val_paths(csv_path, max_per_class=MAX_PER_CLASS):
    import pandas as pd

    df = pd.read_csv(csv_path)

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "path" not in df.columns:
        candidates = [c for c in df.columns if "path" in c or "file" in c or "img" in c]
        if not candidates:
            raise ValueError(f"Cannot find path column in {csv_path}. Columns: {list(df.columns)}")
        df = df.rename(columns={candidates[0]: "path"})

    if "label" not in df.columns:
        candidates = [c for c in df.columns if "label" in c or "fake" in c or "class" in c]
        if not candidates:
            raise ValueError(f"Cannot find label column in {csv_path}. Columns: {list(df.columns)}")
        df = df.rename(columns={candidates[0]: "label"})

    # filter to val split if available
    if "split" in df.columns:
        df = df[df["split"] == "val"].copy()
        print(f"  Found {len(df)} val-split rows in CSV.")
    else:
        # random 20% hold-out by index
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.20
        df = df[mask].copy()
        print(f"  No 'split' column — using random 20% hold-out ({len(df)} rows).")

    # sample up to max_per_class per class
    real = df[df["label"] == 0].sample(min(max_per_class, (df["label"] == 0).sum()), random_state=42)
    fake = df[df["label"] == 1].sample(min(max_per_class, (df["label"] == 1).sum()), random_state=42)
    df   = pd.concat([real, fake]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Calibration set: {(df['label']==0).sum()} real + {(df['label']==1).sum()} fake")
    return list(zip(df["path"].tolist(), df["label"].astype(int).tolist()))


def _collect_logits(predictor, samples, model_name):
    import cv2

    logits_list = []
    labels_list = []
    skipped = 0

    for img_path, label in samples:
        p = Path(img_path)
        if not p.exists():
            skipped += 1
            continue
        try:
            img = cv2.imread(str(p))
            if img is None:
                skipped += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            logit = predictor._run_logit(predictor.models[model_name], img, model_name)
            logits_list.append(logit)
            labels_list.append(label)
        except Exception as e:
            skipped += 1

    if skipped:
        print(f"    Skipped {skipped} images (missing or unreadable).")

    return np.array(logits_list, dtype=np.float64), np.array(labels_list, dtype=np.float64)


def _plot_reliability(ax, probs, labels, title, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(probs[mask].mean())

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.04, alpha=0.7, label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main():
    if not _check_prerequisites():
        return

    from src.inference.predict import DeepfakePredictor
    from src.calibration.temperature_scaling import TemperatureScaler, _ece

    print(f"Loading predictor from {CHECKPOINT_DIR} ...")
    predictor = DeepfakePredictor(str(CHECKPOINT_DIR))

    print(f"Loading val samples from {CSV_PATH} ...")
    try:
        samples = _load_val_paths(CSV_PATH)
    except Exception as e:
        print(f"  ERROR loading CSV: {e}")
        return

    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Reliability Diagrams — Before / After Temperature Scaling")
        has_mpl = True
    except ImportError:
        has_mpl = False
        print("  matplotlib not available — skipping reliability diagrams.")

    results = {}
    for idx, model_name in enumerate(["efb4", "swin", "clip"]):
        print(f"\n[{model_name}] Collecting logits ...")
        logits, labels = _collect_logits(predictor, samples, model_name)

        if len(logits) < 10:
            print(f"  Not enough valid samples ({len(logits)}) — skipping {model_name}.")
            continue

        from src.calibration.temperature_scaling import _sigmoid
        probs_before = _sigmoid(logits)
        ece_before   = _ece(probs_before, labels)

        scaler = TemperatureScaler()
        T      = scaler.fit(logits, labels)

        probs_after = scaler.transform(logits)
        ece_after   = _ece(probs_after, labels)

        print(f"  T={T:.4f}  ECE before={ece_before:.4f}  ECE after={ece_after:.4f}")

        out_path = CALIB_DIR / f"{model_name}_temperature.json"
        scaler.save(str(out_path))
        print(f"  Saved -> {out_path}")

        results[model_name] = {"T": T, "ece_before": ece_before, "ece_after": ece_after}

        if has_mpl:
            _plot_reliability(axes[0, idx], probs_before, labels, f"{model_name} — before (ECE={ece_before:.3f})")
            _plot_reliability(axes[1, idx], probs_after,  labels, f"{model_name} — after  (ECE={ece_after:.3f})")

    if has_mpl and results:
        fig_path = CALIB_DIR / "reliability_diagram.png"
        plt.tight_layout()
        plt.savefig(str(fig_path), dpi=120)
        print(f"\nReliability diagram saved -> {fig_path}")
        plt.close()

    print("\n=== Calibration summary ===")
    for name, r in results.items():
        print(f"  {name:6s}  T={r['T']:.4f}  ECE {r['ece_before']:.4f} -> {r['ece_after']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
