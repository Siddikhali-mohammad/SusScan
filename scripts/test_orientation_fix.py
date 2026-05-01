#!/usr/bin/env python3
"""
Orientation robustness test for SusScan.

Loads a face image, creates 4 rotated versions (0°/90°/180°/270°), runs each
through the full detect → normalize → predict pipeline, and prints a table
showing whether the orientation fix keeps predictions consistent.

Usage:
    python scripts/test_orientation_fix.py [--image path/to/face.jpg]
    python scripts/test_orientation_fix.py --checkpoint-dir checkpoints
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.utils import load_config
from src.inference.predict import DeepfakePredictor
from src.data.face_detector import (
    detect_and_crop_all,
    compute_face_orientation,
    normalize_face_orientation,
)

_ROTATIONS = [
    ("0°  (original)", None),
    ("90° CW",         cv2.ROTATE_90_CLOCKWISE),
    ("180°",           cv2.ROTATE_180),
    ("270° CW",        cv2.ROTATE_90_COUNTERCLOCKWISE),
]


def analyze_rotation(img_rgb, label, predictor):
    print(f"\n  [{label}]  input: {img_rgb.shape[1]}×{img_rgb.shape[0]}")

    faces = detect_and_crop_all(img_rgb, padding=0.30)

    if not faces:
        print("  MTCNN  : no face detected  →  would return HTTP 422")
        return None

    raw_crop, bbox, lm = faces[0]
    print(f"  MTCNN  : face detected  bbox={[int(v) for v in bbox]}")

    if lm is not None:
        ori = compute_face_orientation(lm)
        print(f"  Orient : upside_down={ori['is_upside_down']}  "
              f"tilt={ori['tilt_degrees']:.1f}°  "
              f"too_tilted={ori['is_too_tilted']}")
    else:
        print("  Orient : landmarks unavailable — skipping normalization")

    face_crop, status = normalize_face_orientation(raw_crop, lm)

    if status["rejected"]:
        print(f"  Result : REJECTED — {status['reason']}")
        return None

    if status["rotated_180"]:
        print("  Action : auto-rotated 180°")
    else:
        print("  Action : no rotation applied")

    result = predictor.predict_with_explanation_from_array(face_crop, use_tta=False)
    scores = result["individual_scores"]
    print(f"  Predict: {result['prediction']:12s}  "
          f"ensemble={result['ensemble_score']:.3f}  "
          f"conf={result['confidence']*100:.1f}%")
    print(f"  Scores : efb4={scores['efb4']:.3f}  "
          f"swin={scores['swin']:.3f}  "
          f"clip={scores['clip']:.3f}")
    return result["prediction"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to a face image (JPG/PNG)")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--config",         default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"Loading models from {args.checkpoint_dir} ...")
    predictor = DeepfakePredictor(args.checkpoint_dir, image_size=cfg.data.image_size)

    raw = cv2.imread(args.image)
    if raw is None:
        sys.exit(f"Cannot read image: {args.image}")
    img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    print(f"\nTest image : {args.image}")
    print("=" * 60)

    results = {}
    for label, rot_code in _ROTATIONS:
        rotated = cv2.rotate(img_rgb, rot_code) if rot_code is not None else img_rgb
        results[label] = analyze_rotation(rotated, label, predictor)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("-" * 60)
    baseline = results.get("0°  (original)")
    for label, pred in results.items():
        match = ""
        if label != "0°  (original)" and pred is not None and baseline is not None:
            match = "✓ matches baseline" if pred == baseline else "✗ differs from baseline"
        elif pred is None:
            match = "(no face / rejected)"
        print(f"  {label:<16}  {str(pred):<14}  {match}")
    print()


if __name__ == "__main__":
    main()
