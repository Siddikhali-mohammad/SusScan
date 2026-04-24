#!/usr/bin/env python3
import os
os.environ.setdefault("OPENCV_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2; cv2.setNumThreads(1)
import torch

from src.utils import set_seed, load_config, install_stderr_filter
from src.data import load_master_csv
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD
from src.models.efficientnet import build_efb4
from src.models.swin import build_swin_t
from src.models.clip_vit import build_clip_vit_b16
from src.training.train_efb4 import train_efb4
from src.training.train_swin import train_swin
from src.training.train_clip import train_clip
from src.evaluation.holdout import run_one_holdout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["efb4", "swin", "clip"],
                   choices=["efb4", "swin", "clip"])
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--sequential", action="store_true")
    return p.parse_args()


def main():
    install_stderr_filter()
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    master_df = load_master_csv(cfg.paths.csv_path)
    print(f"loaded {len(master_df):,} rows from {cfg.paths.csv_path}")

    all_cfgs = {
        "efb4": {
            "builder": build_efb4,
            "train_fn": lambda **kw: train_efb4(
                use_compile=not args.no_compile,
                total_epochs=cfg.efb4.total_epochs, swa_window=cfg.efb4.swa_window,
                backbone_lr=cfg.efb4.backbone_lr, head_lr=cfg.efb4.head_lr,
                warmup_epochs=cfg.efb4.warmup_epochs, **kw),
            "normalize_mean": IMAGENET_MEAN, "normalize_std": IMAGENET_STD,
        },
        "swin": {
            "builder": build_swin_t,
            "train_fn": lambda **kw: train_swin(
                total_epochs=cfg.swin.total_epochs, swa_window=cfg.swin.swa_window,
                backbone_lr=cfg.swin.backbone_lr, head_lr=cfg.swin.head_lr,
                warmup_epochs=cfg.swin.warmup_epochs, **kw),
            "normalize_mean": IMAGENET_MEAN, "normalize_std": IMAGENET_STD,
        },
        "clip": {
            "builder": build_clip_vit_b16,
            "train_fn": lambda **kw: train_clip(
                total_epochs=cfg.clip.total_epochs, swa_window=cfg.clip.swa_window,
                backbone_lr=cfg.clip.backbone_lr, head_lr=cfg.clip.head_lr,
                warmup_epochs=cfg.clip.warmup_epochs, **kw),
            "normalize_mean": CLIP_MEAN, "normalize_std": CLIP_STD,
        },
    }

    model_cfgs = {k: v for k, v in all_cfgs.items() if k in args.models}
    print(f"models: {list(model_cfgs.keys())}  holdout: {args.holdout}")

    summary = run_one_holdout(
        holdout_name=args.holdout, master_df=master_df, model_configs=model_cfgs,
        loo_ckpt_dir=cfg.paths.loo_ckpt_dir, loo_log_dir=cfg.paths.loo_log_dir,
        loo_results_dir=cfg.paths.loo_results_dir, device=device,
        image_size=cfg.data.image_size, batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers, val_frac=cfg.data.val_frac,
        seed=cfg.seed, parallel=not args.sequential, meta_C=cfg.meta_learner.C,
    )

    m = summary["ensemble_metrics"]
    if m.get("auroc") is not None:
        print(f"\nAUROC={m['auroc']:.4f}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    else:
        print(f"\nDetection rate: {m['detection_rate']:.4f}")


if __name__ == "__main__":
    main()
