import pickle
from pathlib import Path

import numpy as np
import torch

from .efficientnet import build_efb4
from .swin import build_swin_t
from .clip_vit import build_clip_vit_b16


# must match filenames in checkpoints/ exactly
_CKPTS = {
    "efb4": "best_efb4.pt",
    "swin": "best_swin.pt",
    "clip": "best_clip.pt",
    "meta": "ensemble_meta.pkl",
}


def _load_state(path, device, prefix=None):
    raw = torch.load(path, map_location=device, weights_only=False)
    # checkpoints saved as {"model": state_dict}
    state = raw["model"] if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict) else raw
    # strip torch.compile artifact
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    if prefix:
        state = {f"{prefix}.{k}": v for k, v in state.items()}
    return state


def load_base_models(checkpoint_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(checkpoint_dir)
    builders = {"efb4": build_efb4, "swin": build_swin_t, "clip": build_clip_vit_b16}
    # swin and clip use TimmBinaryWrapper (self.model = timm), so need "model." prefix
    prefixes = {"efb4": None, "swin": "model", "clip": "model"}

    models = {}
    for key, fn in builders.items():
        m, _, _ = fn(pretrained=False)
        m.load_state_dict(_load_state(ckpt_dir / _CKPTS[key], device, prefix=prefixes[key]))
        models[key] = m.to(device).eval()
        print(f"  loaded {key}: {_CKPTS[key]}")

    return models


def load_meta_learner(checkpoint_dir):
    path = Path(checkpoint_dir) / _CKPTS["meta"]
    with open(path, "rb") as f:
        meta = pickle.load(f)
    print(f"  loaded meta-learner: {path.name}")
    return meta


@torch.no_grad()
def predict_ensemble(models, meta_learner, images, device=None, amp_dtype=torch.bfloat16):
    if device is None:
        device = next(iter(models.values())).parameters().__next__().device

    images = images.to(device, non_blocking=True)
    from torch.cuda.amp import autocast

    probs = []
    for key in ("efb4", "swin", "clip"):
        models[key].eval()
        with autocast(dtype=amp_dtype):
            logits = models[key](images)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())

    # order matters — must match meta-learner training order
    return meta_learner.predict_proba(np.stack(probs, axis=1))[:, 1]
