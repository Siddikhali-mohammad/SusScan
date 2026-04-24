#!/usr/bin/env python3
import os
os.environ.setdefault("OPENCV_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2; cv2.setNumThreads(1)
import numpy as np

from src.utils import set_seed, load_config
from src.inference.predict import DeepfakePredictor
from src.inference.visualize import build_combined_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg.seed)

    predictor = DeepfakePredictor(args.checkpoint_dir, image_size=cfg.data.image_size)
    result    = predictor.predict_with_explanation(
        args.image, use_tta=not args.no_tta, threshold=args.threshold
    )

    print(f"\nimage    : {args.image}")
    print(f"decision : {result['prediction']}  (confidence: {result['confidence']*100:.1f}%)")
    print(f"ensemble : {result['ensemble_score']:.4f}  (t={args.threshold:.2f})")
    for name, p in result["individual_scores"].items():
        print(f"  {name}: {p:.4f}")
    print(f"\ntop regions:")
    for r in result["top_regions"]:
        print(f"  {r['label']}: {r['intensity']:.4f}")
    print(f"\nexplanation:\n{result['explanation']}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "visualization.png"
        path.write_bytes(build_combined_png(result))
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
