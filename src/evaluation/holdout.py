import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ..data.dataset import DeepfakeDataset, build_holdout_loaders
from ..data.transforms import build_tta_augs, IMAGENET_MEAN, IMAGENET_STD
from ..training.trainer import load_trained_model
from .metrics import compute_holdout_metrics


@torch.no_grad()
def collect_predictions(model, loader, device, return_indices=False, amp_dtype=torch.bfloat16):
    model.eval()
    probs_all, labels_all, idx_all = [], [], []

    for batch in loader:
        if len(batch) == 3:
            imgs, lbls, idx = batch
            idx_all.append(idx.numpy())
        else:
            imgs, lbls = batch

        with autocast(dtype=amp_dtype):
            logits = model(imgs.to(device, non_blocking=True))

        probs_all.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels_all.append(lbls.numpy())

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)

    if return_indices:
        idxs = np.concatenate(idx_all) if idx_all else np.arange(len(labels))
        return probs, labels, idxs
    return probs, labels


@torch.no_grad()
def tta_predict(model, df, mean, std, device, image_size=224,
                batch_size=192, num_workers=12, amp_dtype=torch.bfloat16):
    model.eval()
    prob_sum = None
    final_labels = final_idx = None

    for tfm in build_tta_augs(image_size, mean, std):
        loader = DataLoader(
            DeepfakeDataset(df, tfm, return_index=True),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        p, l, idx = collect_predictions(model, loader, device, return_indices=True, amp_dtype=amp_dtype)

        if prob_sum is None:
            prob_sum, final_labels, final_idx = p.copy(), l, idx
        else:
            prob_sum += p

    return prob_sum / 4, final_labels, final_idx


def run_one_holdout(holdout_name, master_df, model_configs, loo_ckpt_dir, loo_log_dir,
                    loo_results_dir, device, image_size=224, batch_size=96,
                    num_workers=12, val_frac=0.20, seed=42, parallel=True,
                    meta_C=1.0, amp_dtype=torch.bfloat16):
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  HOLDOUT: {holdout_name}")
    print(f"{'='*60}")

    ckpt_dir = Path(loo_ckpt_dir) / f"holdout_{holdout_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(loo_results_dir) / f"holdout_{holdout_name}"
    res_dir.mkdir(parents=True, exist_ok=True)

    first = next(iter(model_configs.values()))
    norm_mean = first.get("normalize_mean", IMAGENET_MEAN)
    norm_std  = first.get("normalize_std", IMAGENET_STD)

    print("\nbuilding loaders...")
    t0 = time.time()
    loaders = build_holdout_loaders(master_df, holdout_name, image_size=image_size,
                                    batch_size=batch_size, num_workers=num_workers,
                                    val_frac=val_frac, seed=seed,
                                    normalize_mean=norm_mean, normalize_std=norm_std)
    print(f"  train={len(loaders['train_df']):,}  val={len(loaders['val_df']):,}"
          f"  test={len(loaders['test_df']):,}  ({time.time()-t0:.1f}s)")

    print("\ntraining models...")
    model_results = {}

    def train_one(name, cfg):
        model_results[name] = cfg["train_fn"](
            train_loader=loaders["train_loader"], val_loader=loaders["val_loader"],
            holdout_name=holdout_name, ckpt_dir=loo_ckpt_dir, log_dir=loo_log_dir, device=device,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if parallel and len(model_configs) > 1:
        with ThreadPoolExecutor(max_workers=len(model_configs)) as pool:
            futures = [pool.submit(train_one, n, c) for n, c in model_configs.items()]
            for fut in futures:
                fut.result()
    else:
        for name, cfg in model_configs.items():
            train_one(name, cfg)

    print("\ngetting val predictions for meta-learner...")
    val_probs = {}
    val_labels = None

    for name, res in model_results.items():
        cfg = model_configs[name]
        print(f"  {name} ({res['winner']}): {Path(res['winner_path']).name}")
        m = load_trained_model(cfg["builder"], res["winner_path"], device)
        p, l = collect_predictions(m, loaders["val_loader"], device, amp_dtype=amp_dtype)
        val_probs[name] = p
        if val_labels is None:
            val_labels = l
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_names = list(model_configs.keys())
    val_probs_arr = np.stack([val_probs[n] for n in model_names], axis=1)
    val_feats = np.column_stack([
        val_probs_arr,
        val_probs_arr.mean(axis=1),
        val_probs_arr.std(axis=1),
        val_probs_arr.max(axis=1) - val_probs_arr.min(axis=1),
    ])

    scaler = StandardScaler()
    val_feats_scaled = scaler.fit_transform(val_feats)

    print(f"\nfitting meta-learner on {len(val_labels):,} val samples...")
    meta = LogisticRegression(C=meta_C, max_iter=1000, class_weight="balanced",
                               solver="lbfgs", random_state=seed)
    meta.fit(val_feats_scaled, val_labels)

    coefs = meta.coef_[0]
    for name, coef in zip(model_names, coefs[:3]):
        print(f"  {name}: {coef:+.3f}")
    print(f"  intercept: {meta.intercept_[0]:+.3f}")

    val_ens_probs = meta.predict_proba(val_feats_scaled)[:, 1]
    val_auroc = roc_auc_score(val_labels, val_ens_probs)
    print(f"  val AUROC: {val_auroc:.4f}")

    with open(ckpt_dir / "meta_learner.pkl", "wb") as f:
        pickle.dump({"meta": meta, "scaler": scaler}, f)

    test_df = loaders["test_df"]
    print(f"\nrunning TTA on {len(test_df):,} test images...")
    test_probs = {}
    test_labels = test_idx = None

    for name, res in model_results.items():
        cfg = model_configs[name]
        print(f"  TTA: {name}")
        m = load_trained_model(cfg["builder"], res["winner_path"], device)
        p, l, idx = tta_predict(
            m, test_df,
            cfg.get("normalize_mean", IMAGENET_MEAN),
            cfg.get("normalize_std", IMAGENET_STD),
            device, image_size=image_size,
            batch_size=batch_size * 2, num_workers=num_workers, amp_dtype=amp_dtype,
        )
        test_probs[name] = p
        if test_labels is None:
            test_labels, test_idx = l, idx
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_probs_arr = np.stack([test_probs[n] for n in model_names], axis=1)
    test_feats = np.column_stack([
        test_probs_arr,
        test_probs_arr.mean(axis=1),
        test_probs_arr.std(axis=1),
        test_probs_arr.max(axis=1) - test_probs_arr.min(axis=1),
    ])
    ens_probs = meta.predict_proba(scaler.transform(test_feats))[:, 1]

    # compute and print final metrics
    model_metrics = {n: compute_holdout_metrics(test_probs[n], test_labels, holdout_name) for n in model_names}
    ens_metrics   = compute_holdout_metrics(ens_probs, test_labels, holdout_name)

    print(f"\nresults for {holdout_name}:")
    for name, m in model_metrics.items():
        if m.get("auroc") is not None:
            print(f"  {name}: AUROC={m['auroc']:.4f}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
        else:
            print(f"  {name}: detection_rate={m['detection_rate']:.4f}")
    if ens_metrics.get("auroc") is not None:
        print(f"  ensemble: AUROC={ens_metrics['auroc']:.4f}  acc={ens_metrics['accuracy']:.4f}")
    else:
        print(f"  ensemble: detection_rate={ens_metrics['detection_rate']:.4f}")

    # dump per-image predictions so we can analyze later
    pd.DataFrame({
        "idx":            test_idx,
        "path":           [test_df["path"].iloc[int(i)] for i in test_idx],
        "true_label":     test_labels.astype(int),
        "ensemble_prob":  ens_probs,
        **{f"{n}_prob": test_probs[n] for n in model_names},
    }).to_parquet(res_dir / "per_image_predictions.parquet", index=False)

    elapsed = time.time() - t_start
    summary = {
        "holdout_name":    holdout_name,
        "total_time_sec":  elapsed,
        "total_time_min":  elapsed / 60,
        "train_size":      len(loaders["train_df"]),
        "val_size":        len(loaders["val_df"]),
        "test_size":       len(test_df),
        "per_model_metrics": model_metrics,
        "ensemble_metrics":  ens_metrics,
        "meta_learner": {
            "coefficients": coefs.tolist(),
            "intercept":    float(meta.intercept_[0]),
            "val_auroc":    float(val_auroc),
            "model_order":  model_names,
        },
        "model_winners": {n: r["winner"] for n, r in model_results.items()},
    }

    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\ndone in {elapsed/60:.1f} min")
    return summary


