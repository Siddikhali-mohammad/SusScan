#!/usr/bin/env python3
import os
os.environ.setdefault("OPENCV_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse, json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2; cv2.setNumThreads(1)
import numpy as np
import torch

from src.utils import set_seed, load_config, install_stderr_filter
from src.data import load_master_csv, build_holdout_loaders
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD
from src.models import load_base_models, load_meta_learner
from src.evaluation.holdout import tta_predict
from src.evaluation.metrics import compute_holdout_metrics, print_metrics_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    install_stderr_filter()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    loaders = build_holdout_loaders(
        load_master_csv(args.csv), args.holdout,
        image_size=cfg.data.image_size, batch_size=cfg.data.batch_size * 2,
        num_workers=cfg.data.num_workers, val_frac=cfg.data.val_frac, seed=cfg.seed,
    )
    test_df = loaders["test_df"]
    print(f"test set: {len(test_df):,} images from '{args.holdout}'")

    models = load_base_models(args.checkpoint_dir, device)
    meta   = load_meta_learner(args.checkpoint_dir)

    norms = {"efb4": (IMAGENET_MEAN, IMAGENET_STD),
             "swin": (IMAGENET_MEAN, IMAGENET_STD),
             "clip": (CLIP_MEAN, CLIP_STD)}

    test_probs, test_labels, test_idx = {}, None, None

    for name, model in models.items():
        mean, std = norms[name]
        print(f"\n  {name}...")

        if args.no_tta:
            from src.evaluation.holdout import collect_predictions
            from src.data import DeepfakeDataset
            from src.data.transforms import build_val_augs
            from torch.utils.data import DataLoader
            ds = DeepfakeDataset(test_df, build_val_augs(cfg.data.image_size, mean, std), return_index=True)
            loader = DataLoader(ds, batch_size=cfg.data.batch_size*2, shuffle=False,
                                num_workers=cfg.data.num_workers, pin_memory=True)
            p, l, idx = collect_predictions(model, loader, device, return_indices=True)
        else:
            p, l, idx = tta_predict(model, test_df, mean, std, device,
                                    image_size=cfg.data.image_size,
                                    batch_size=cfg.data.batch_size*2,
                                    num_workers=cfg.data.num_workers)

        test_probs[name] = p
        if test_labels is None: test_labels, test_idx = l, idx
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    ens = meta.predict_proba(np.stack([test_probs[n] for n in ["efb4", "swin", "clip"]], axis=1))[:, 1]

    metrics = {n: compute_holdout_metrics(p, test_labels, args.holdout) for n, p in test_probs.items()}
    metrics["ensemble"] = compute_holdout_metrics(ens, test_labels, args.holdout)
    print_metrics_table(metrics)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(out, "w"), indent=2)
        print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
