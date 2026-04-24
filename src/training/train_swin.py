from pathlib import Path
import torch

from ..models.swin import build_swin_t
from .trainer import train_one_model


def train_swin(train_loader, val_loader, holdout_name, ckpt_dir, log_dir,
               device=None, total_epochs=10, swa_window=3,
               backbone_lr=2e-5, head_lr=1e-3, warmup_epochs=1,
               label_smoothing=0.05, weight_decay=0.01, grad_clip_norm=1.0,
               use_compile=False, seed=42):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return train_one_model(
        "swin_t", build_swin_t, train_loader, val_loader,
        total_epochs=total_epochs, swa_window=swa_window,
        backbone_lr=backbone_lr, head_lr=head_lr, warmup_epochs=warmup_epochs,
        holdout_name=holdout_name, ckpt_dir=Path(ckpt_dir), log_dir=Path(log_dir),
        device=device, label_smoothing=label_smoothing, weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, use_compile=use_compile, seed=seed,
    )
