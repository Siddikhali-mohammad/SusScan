#!/usr/bin/env python3
import os
os.environ.setdefault("OPENCV_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse, pickle, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2; cv2.setNumThreads(1)
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.utils import set_seed, load_config
from src.data import load_master_csv, build_holdout_loaders
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.efficientnet import build_efb4
from src.models.swin import build_swin_t
from src.models.clip_vit import build_clip_vit_b16
from src.training.trainer import load_trained_model
from src.evaluation.holdout import collect_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)

    loaders = build_holdout_loaders(
        load_master_csv(args.csv), args.holdout,
        image_size=cfg.data.image_size, batch_size=cfg.data.batch_size * 2,
        num_workers=cfg.data.num_workers, val_frac=cfg.data.val_frac, seed=cfg.seed,
        normalize_mean=IMAGENET_MEAN, normalize_std=IMAGENET_STD,
    )

    specs = [
        ("efb4", build_efb4,         ckpt_dir / "best_efb4.pt"),
        ("swin", build_swin_t,       ckpt_dir / "best_swin.pt"),
        ("clip", build_clip_vit_b16, ckpt_dir / "best_clip.pt"),
    ]

    val_probs, val_labels = {}, None
    for name, builder, ckpt in specs:
        print(f"  loading {name}...")
        m = load_trained_model(builder, str(ckpt), device)
        p, l = collect_predictions(m, loaders["val_loader"], device)
        val_probs[name] = p
        if val_labels is None: val_labels = l
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    probs_arr = np.stack([val_probs["efb4"], val_probs["swin"], val_probs["clip"]], axis=1)
    feats = np.column_stack([
        probs_arr,
        probs_arr.mean(axis=1),
        probs_arr.std(axis=1),
        probs_arr.max(axis=1) - probs_arr.min(axis=1),
    ])

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    print(f"\nfitting on {len(val_labels):,} samples...")
    meta = LogisticRegression(C=cfg.meta_learner.C, max_iter=cfg.meta_learner.max_iter,
                               class_weight=cfg.meta_learner.class_weight,
                               solver=cfg.meta_learner.solver, random_state=cfg.seed)
    meta.fit(feats_scaled, val_labels)

    c = meta.coef_[0]
    print(f"  efb4={c[0]:+.4f}  swin={c[1]:+.4f}  clip={c[2]:+.4f}  intercept={meta.intercept_[0]:+.4f}")
    print(f"  val AUROC: {roc_auc_score(val_labels, meta.predict_proba(feats_scaled)[:, 1]):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"meta": meta, "scaler": scaler}, open(out, "wb"))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
