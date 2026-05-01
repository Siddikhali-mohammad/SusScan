import pickle
from pathlib import Path

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


