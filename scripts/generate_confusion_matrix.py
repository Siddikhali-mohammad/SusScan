import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

RUTGERS_SCARLET = "#CC0033"
OFF_DIAG_COLOR  = "#2B2B2B"
BG_COLOR        = "#F9F9F9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="data/loo_sweep_v2/results/holdout_140k_clean_v1/per_image_predictions.parquet")
    ap.add_argument("--out", default="results/confusion_matrix_140k.png")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df    = pd.read_parquet(args.parquet)
    y     = df["true_label"].values
    probs = df["ensemble_prob"].values
    preds = (probs >= args.threshold).astype(int)

    cm    = confusion_matrix(y, preds)
    auroc = roc_auc_score(y, probs)
    acc   = accuracy_score(y, preds)
    n     = len(y)

    tn, fp, fn, tp = cm.ravel()

    # Build annotation: "count\nX.X%"
    annot = np.array([
        [f"{tn}\n{tn/n*100:.1f}%", f"{fp}\n{fp/n*100:.1f}%"],
        [f"{fn}\n{fn/n*100:.1f}%", f"{tp}\n{tp/n*100:.1f}%"],
    ])

    # Custom color matrix: diagonal = scarlet, off-diagonal = dark gray
    color_vals = np.array([[0.0, 1.0], [1.0, 0.0]])  # 0 → off-diag, 1 → diag

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw cells manually for dual-color control
    from matplotlib.colors import LinearSegmentedColormap
    diag_cmap   = LinearSegmentedColormap.from_list("diag",   ["#fce4ec", RUTGERS_SCARLET])
    offdiag_cmap= LinearSegmentedColormap.from_list("offdiag",["#f5f5f5", OFF_DIAG_COLOR])

    cell_colors = np.array([
        [RUTGERS_SCARLET, OFF_DIAG_COLOR],
        [OFF_DIAG_COLOR,  RUTGERS_SCARLET],
    ])
    text_colors = [["white", "white"], ["white", "white"]]

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j, 1 - i), 1, 1,
                color=cell_colors[i][j], zorder=1
            ))
            ax.text(j + 0.5, 1.5 - i, annot[i][j],
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color=text_colors[i][j], zorder=2)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["REAL (pred)", "FAKE (pred)"], fontsize=12)
    ax.set_yticklabels(["FAKE (true)", "REAL (true)"], fontsize=12)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)

    ax.set_title(
        f"Confusion Matrix — 140k StyleGAN LOO Fold\n"
        f"Ensemble AUROC: {auroc:.4f}   Accuracy: {acc:.4f}   N={n:,}",
        fontsize=13, pad=20, color="#111111"
    )

    # Draw grid lines
    for x in range(1, 2):
        ax.axvline(x, color="white", linewidth=3, zorder=3)
    for y_ in range(1, 2):
        ax.axhline(y_, color="white", linewidth=3, zorder=3)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved: {out.resolve()}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  AUROC={auroc:.4f}  ACC={acc:.4f}")


if __name__ == "__main__":
    main()
