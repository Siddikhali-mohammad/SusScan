import json
import numpy as np
from pathlib import Path


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        def nll(T):
            p = _sigmoid(logits / T)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

        # Golden-section search over T in [0.1, 10.0]
        a, b = 0.1, 10.0
        gr = (np.sqrt(5) + 1) / 2
        for _ in range(200):
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            if nll(c) < nll(d):
                b = d
            else:
                a = c
        self.T = (a + b) / 2
        return self.T

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(logits, dtype=np.float64) / self.T)

    def calibrate_prob(self, prob: float) -> float:
        prob = float(np.clip(prob, 1e-9, 1 - 1e-9))
        logit = float(np.log(prob / (1 - prob)))
        return float(self.transform(np.array([logit]))[0])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"T": self.T}, f, indent=2)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path) as f:
            obj.T = float(json.load(f)["T"])
        return obj
