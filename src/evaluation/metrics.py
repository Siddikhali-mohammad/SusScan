import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from ..data.dataset import SINGLE_CLASS_DATASETS


def compute_holdout_metrics(probs, labels, holdout_name, threshold=0.5):
    single_class = holdout_name in SINGLE_CLASS_DATASETS
    preds = (probs > threshold).astype(int)

    out = {
        "holdout_name": holdout_name,
        "n_samples": len(labels),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "is_single_class": single_class,
        "mean_prob": float(probs.mean()),
        "std_prob": float(probs.std()),
        "threshold": threshold,
    }

    if single_class:
        out.update(detection_rate=float((preds == 1).mean()),
                   mean_fake_prob=float(probs.mean()),
                   auroc=None, accuracy=None, f1=None)
    else:
        out["accuracy"] = float(accuracy_score(labels, preds))
        out["auroc"] = float(roc_auc_score(labels, probs))
        out["f1"] = float(f1_score(labels, preds))
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                   precision=float(tp / max(tp + fp, 1)),
                   recall=float(tp / max(tp + fn, 1)))

    return out


def find_optimal_threshold(probs, labels, metric="f1"):
    fpr, tpr, thresholds = roc_curve(labels, probs)

    if metric == "youden":
        idx = int(np.argmax(tpr - fpr))
    elif metric == "f1":
        idx, best = 0, -1.0
        for i, t in enumerate(thresholds):
            f = f1_score(labels, (probs >= t).astype(int), zero_division=0)
            if f > best:
                best, idx = f, i
    else:
        raise ValueError(f"metric must be 'f1' or 'youden', got {metric!r}")

    return float(thresholds[idx])


def print_metrics_table(results):
    print("\n" + "=" * 80)
    print(f"{'Holdout':<35}  {'AUROC':>7}  {'Acc':>6}  {'F1':>6}  {'Det.Rate':>9}")
    print("-" * 80)
    for name, m in results.items():
        auroc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "  N/A  "
        acc   = f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "  N/A "
        f1    = f"{m['f1']:.4f}" if m.get("f1") is not None else "  N/A "
        det   = f"{m['detection_rate']:.4f}" if m.get("detection_rate") is not None else "  N/A   "
        print(f"  {name:<33}  {auroc:>7}  {acc:>6}  {f1:>6}  {det:>9}")
    print("=" * 80)
