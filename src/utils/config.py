from pathlib import Path
import yaml


class _Namespace:
    # lets us do cfg.efb4.backbone_lr instead of cfg["efb4"]["backbone_lr"]
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return f"Namespace({vars(self)})"

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(path=None):
    if path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        path = repo_root / "configs" / "default.yaml"
    with open(path) as f:
        return _Namespace(yaml.safe_load(f))
