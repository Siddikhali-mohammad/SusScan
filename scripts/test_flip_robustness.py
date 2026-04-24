import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.predict import DeepfakePredictor


TOLERANCE = 0.02   # max allowed ensemble diff between original and flip
N_IMAGES   = 5     # real faces to test


def _pick_real_images(csv_path: str, n: int = N_IMAGES) -> list[str]:
    df = pd.read_csv(csv_path)
    real = df[df["label"] == 0].dropna(subset=["path"])
    paths = real["path"].tolist()
    # Walk until we find n images that actually exist on disk
    found = []
    for p in paths:
        if Path(p).exists():
            found.append(p)
            if len(found) >= n:
                break
    if not found:
        raise RuntimeError(
            f"Could not find any real images from {csv_path}. "
            "Check that data/master_v4.csv exists and image paths in it are valid."
        )
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--csv",            default="data/master_v4.csv")
    args = ap.parse_args()

    predictor = DeepfakePredictor(args.checkpoint_dir)
    paths = _pick_real_images(args.csv)
    print(f"\nTesting flip robustness on {len(paths)} real images")
    print(f"{'Image':<40} {'Orig ens':>9} {'Flip ens':>9} {'Delta':>8} | "
          f"{'efb4 Δ':>8} {'swin Δ':>8} {'clip Δ':>8}")
    print("-" * 100)

    all_pass = True
    worst_model = {"efb4": 0.0, "swin": 0.0, "clip": 0.0}

    for path in paths:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img_flip = cv2.flip(img, 1)

        # Score with the FIX in place (use_tta=False; flip averaging is now built in)
        # To reproduce the BUG, temporarily use only t_in = [val(original)] by
        # passing a pre-flipped image and comparing — the scores should now be close.
        result_orig = predictor.predict_from_array(img,      use_tta=False)
        result_flip = predictor.predict_from_array(img_flip, use_tta=False)

        ens_orig  = result_orig["fake_prob"]
        ens_flip  = result_flip["fake_prob"]
        delta_ens = abs(ens_orig - ens_flip)

        mp_o = result_orig["model_probs"]
        mp_f = result_flip["model_probs"]
        d_efb4 = abs(mp_o["efb4"] - mp_f["efb4"])
        d_swin = abs(mp_o["swin"] - mp_f["swin"])
        d_clip = abs(mp_o["clip"] - mp_f["clip"])

        worst_model["efb4"] = max(worst_model["efb4"], d_efb4)
        worst_model["swin"] = max(worst_model["swin"], d_swin)
        worst_model["clip"] = max(worst_model["clip"], d_clip)

        status = "OK" if delta_ens <= TOLERANCE else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{Path(path).name:<40} {ens_orig:>9.4f} {ens_flip:>9.4f} "
              f"{delta_ens:>7.4f}{'*' if status=='FAIL' else ' '} | "
              f"{d_efb4:>8.4f} {d_swin:>8.4f} {d_clip:>8.4f}  {status}")

    print()
    worst = max(worst_model, key=worst_model.get)
    print(f"Worst offender model (max per-image delta): {worst} "
          f"(Δ={worst_model[worst]:.4f})")
    for m, v in worst_model.items():
        print(f"  {m}: max_delta={v:.4f}")

    print()
    if all_pass:
        print(f"PASS — all {len(paths)} images within tolerance {TOLERANCE}")
    else:
        print(f"FAIL — some images exceeded tolerance {TOLERANCE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
