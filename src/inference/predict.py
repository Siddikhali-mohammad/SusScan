from pathlib import Path

import cv2
import numpy as np
import torch

from ..data.transforms import (
    build_val_augs, build_tta_augs,
    IMAGENET_MEAN, IMAGENET_STD, CLIP_MEAN, CLIP_STD,
)
from ..models.ensemble import load_base_models, load_meta_learner
from ..explainability.gradcam import get_gradcam, overlay_heatmap
from ..explainability.region_mapping import map_regions
from ..explainability.explain import generate_explanation, model_agreement, build_story
from ..classification.verdict import classify, Verdict
from ..calibration.temperature_scaling import TemperatureScaler


# SUPERSEDED by src/classification/verdict.py — kept for A/B comparison
# Decision thresholds.
# prob < LOW                          → REAL
# LOW ≤ prob ≤ HIGH                   → UNCERTAIN
# prob > HIGH AND base majority agree → FAKE
# prob > HIGH BUT base models split   → UNCERTAIN (CLIP-dominated false positive)
# _FAKE_THRESHOLD_HIGH = 0.62
# _FAKE_THRESHOLD_LOW  = 0.40
#
#
# def _classify(prob, individual_scores=None):
#     """
#     Three-state classifier with a base-model majority veto.
#
#     The meta-learner gives CLIP 2.34× weight and carries a +0.687 intercept
#     (baseline ~66% fake even when all models are uncertain). This means CLIP alone
#     can push the ensemble above the FAKE threshold even when EFB4 and Swin both
#     say REAL. The veto ensures a FAKE verdict requires at least 2/3 base models
#     to individually agree (prob > 0.5) — preventing single-model dominance.
#     """
#     if prob <= _FAKE_THRESHOLD_LOW:
#         return "REAL", 1.0 - prob
#
#     if prob >= _FAKE_THRESHOLD_HIGH:
#         if individual_scores is not None:
#             fake_votes = sum(1 for s in individual_scores.values() if s >= 0.5)
#             if fake_votes >= 2:
#                 return "FAKE", prob
#             # Meta-learner fired but base models don't agree — likely CLIP-dominated
#             return "UNCERTAIN", prob
#
#         return "FAKE", prob   # fallback if no individual scores provided
#
#     return "UNCERTAIN", prob


def _load_calibration(checkpoint_dir):
    scalers = {}
    calib_dir = Path(checkpoint_dir) / "calibration"
    for name in ("efb4", "swin", "clip"):
        path = calib_dir / f"{name}_temperature.json"
        if path.exists():
            try:
                scalers[name] = TemperatureScaler.load(str(path))
                print(f"  [calibration] {name}: T={scalers[name].T:.4f}")
            except Exception as e:
                print(f"  WARNING: could not load calibration for {name}: {e} — using T=1.0")
                scalers[name] = TemperatureScaler()
        else:
            print(f"  WARNING: calibration file not found for {name} ({path}) — using T=1.0 (no-op)")
            scalers[name] = TemperatureScaler()
    return scalers


class DeepfakePredictor:
    def __init__(self, checkpoint_dir, device=None, amp_dtype=torch.bfloat16, image_size=224):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.amp_dtype = amp_dtype
        self.image_size = image_size

        print(f"loading ensemble from {checkpoint_dir} on {device}")
        self.models = load_base_models(checkpoint_dir, device)
        self.meta = load_meta_learner(checkpoint_dir)

        # pre-build transforms — CLIP uses different stats than EFB4/Swin
        self._val_in  = build_val_augs(image_size, IMAGENET_MEAN, IMAGENET_STD)
        self._val_cl  = build_val_augs(image_size, CLIP_MEAN, CLIP_STD)
        self._tta_in  = build_tta_augs(image_size, IMAGENET_MEAN, IMAGENET_STD)
        self._tta_cl  = build_tta_augs(image_size, CLIP_MEAN, CLIP_STD)

        # load temperature scalers (falls back to T=1.0 if not found)
        self._scalers = _load_calibration(checkpoint_dir)

    def _read(self, path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"not found: {path}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"could not decode: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def _run(self, model, tensors):
        from torch.cuda.amp import autocast
        model.eval()
        ps = []
        for t in tensors:
            with autocast(dtype=self.amp_dtype):
                logit = model(t.unsqueeze(0).to(self.device))
            ps.append(torch.sigmoid(logit.float()).item())
        return float(np.mean(ps))

    @torch.no_grad()
    def _run_logit(self, model, img_rgb, model_name):
        # returns pre-sigmoid logit — used by calibrate_models.py for temperature fitting
        from torch.cuda.amp import autocast
        model.eval()
        aug = self._val_cl if model_name == "clip" else self._val_in
        t   = aug(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        with autocast(dtype=self.amp_dtype):
            logit = model(t)
        return logit.float().item()

    def _score_img(self, img, use_tta):
        if use_tta:
            t_in = [a(image=img)["image"] for a in self._tta_in]
            t_cl = [a(image=img)["image"] for a in self._tta_cl]
        else:
            # Average original + horizontal flip so orientation bias in training data
            # (most faces photographed right-way-up) does not flip the verdict.
            # Full TTA already includes the flip as view 2 of 4; only add it here.
            img_flip = cv2.flip(img, 1)
            t_in = [self._val_in(image=img)["image"], self._val_in(image=img_flip)["image"]]
            t_cl = [self._val_cl(image=img)["image"], self._val_cl(image=img_flip)["image"]]

        logits = {}
        with torch.no_grad():
            from torch.cuda.amp import autocast
            for name, tensors in [("efb4", t_in), ("swin", t_in), ("clip", t_cl)]:
                self.models[name].eval()
                raw_logits = []
                for t in tensors:
                    with autocast(dtype=self.amp_dtype):
                        logit = self.models[name](t.unsqueeze(0).to(self.device))
                    raw_logits.append(logit.float().item())
                logits[name] = float(np.mean(raw_logits))

        probs = {
            name: float(self._scalers[name].transform(np.array([logits[name]]))[0])
            for name in ("efb4", "swin", "clip")
        }
        return probs

    def _meta_predict(self, model_probs):
        p  = np.array([model_probs["efb4"], model_probs["swin"], model_probs["clip"]])
        x  = np.array([[*p, p.mean(), p.std(), p.max() - p.min()]])
        lr = self.meta["meta"]
        xs = self.meta["scaler"].transform(x)
        logit = float(xs @ lr.coef_.T + lr.intercept_)
        prob  = float(1 / (1 + np.exp(-logit)))

        vr = classify(
            efb4_prob=model_probs["efb4"],
            swin_prob=model_probs["swin"],
            clip_prob=model_probs["clip"],
            ensemble_prob=prob,
        )

        print(
            f"  probs=efb4:{model_probs['efb4']:.2f} swin:{model_probs['swin']:.2f} "
            f"clip:{model_probs['clip']:.2f} ens={prob:.2f} | "
            f"spread={vr.spread:.2f} margin={vr.margin:.2f} | "
            f"rule={vr.triggered_rule} | {vr.verdict.value}"
        )

        return vr

    def predict(self, image_path, use_tta=True, threshold=0.5):
        img = self._read(image_path)
        model_probs = self._score_img(img, use_tta)
        vr = self._meta_predict(model_probs)
        return {
            "fake_prob":   vr.ensemble_prob,
            "prediction":  vr.verdict.value,
            "confidence":  (1.0 - vr.ensemble_prob) if vr.verdict in (Verdict.LIKELY_REAL, Verdict.STRONG_REAL) else vr.ensemble_prob,
            "is_fake":     vr.verdict in (Verdict.LIKELY_FAKE, Verdict.STRONG_FAKE),
            "model_probs": model_probs,
        }

    def predict_from_array(self, img_rgb, use_tta=False):
        model_probs = self._score_img(img_rgb, use_tta)
        vr = self._meta_predict(model_probs)
        return {
            "fake_prob":   vr.ensemble_prob,
            "prediction":  vr.verdict.value,
            "confidence":  (1.0 - vr.ensemble_prob) if vr.verdict in (Verdict.LIKELY_REAL, Verdict.STRONG_REAL) else vr.ensemble_prob,
            "is_fake":     vr.verdict in (Verdict.LIKELY_FAKE, Verdict.STRONG_FAKE),
            "model_probs": model_probs,
        }

    def predict_with_explanation(self, image_path, use_tta=True, threshold=0.5):
        img         = self._read(image_path)
        model_probs = self._score_img(img, use_tta)
        vr          = self._meta_predict(model_probs)
        return self._explain_from_img(img, model_probs, vr)

    def _explain_from_img(self, img, scores, vr):
        # single-pass tensors for GradCAM — no TTA, gradients needed
        t_in = self._val_in(image=img)["image"].unsqueeze(0)
        t_cl = self._val_cl(image=img)["image"].unsqueeze(0)

        heatmaps = {}
        for name in ("efb4", "swin", "clip"):
            tensor = t_cl if name == "clip" else t_in
            try:
                heatmaps[name] = get_gradcam(self.models[name], name, tensor, self.device)
            except Exception as e:
                print(f"  GradCAM failed for {name}: {e}")
                heatmaps[name] = np.zeros((224, 224), dtype=np.float32)

        img_224     = cv2.resize(img, (self.image_size, self.image_size))
        overlays    = {name: overlay_heatmap(img_224, cam) for name, cam in heatmaps.items()}
        avg_cam     = np.mean(list(heatmaps.values()), axis=0)
        top_regions = map_regions(avg_cam)

        verdict_str_3 = _verdict_to_3state(vr.verdict)
        agreement   = model_agreement(scores, verdict_str_3)
        explanation = generate_explanation(vr.verdict.value, vr.ensemble_prob if vr.verdict in (Verdict.LIKELY_FAKE, Verdict.STRONG_FAKE) else 1 - vr.ensemble_prob, scores, top_regions, agreement)
        story       = build_story(
            vr.verdict.value,
            vr.ensemble_prob if vr.verdict in (Verdict.LIKELY_FAKE, Verdict.STRONG_FAKE) else 1 - vr.ensemble_prob,
            scores,
            top_regions,
            agreement,
            reason=vr.reason,
        )

        confidence = (1.0 - vr.ensemble_prob) if vr.verdict in (Verdict.LIKELY_REAL, Verdict.STRONG_REAL) else vr.ensemble_prob

        return {
            "prediction":        vr.verdict.value,
            "ui_color":          vr.ui_color,
            "ui_label":          vr.ui_label,
            "ui_muted":          vr.ui_muted,
            "reason":            vr.reason,
            "triggered_rule":    vr.triggered_rule,
            "confidence":        confidence,
            "individual_scores": scores,
            "ensemble_score":    vr.ensemble_prob,
            "spread":            vr.spread,
            "std":               vr.std,
            "margin":            vr.margin,
            "votes_fake":        vr.votes_fake,
            "votes_real":        vr.votes_real,
            "heatmaps":          heatmaps,
            "overlay_images":    overlays,
            "original_image":    img_224,
            "top_regions":       top_regions,
            "explanation":       explanation,
            "story":             story,
            "model_agreement":   agreement,
        }

    def predict_with_explanation_from_array(self, img_rgb, use_tta=False):
        model_probs = self._score_img(img_rgb, use_tta)
        vr          = self._meta_predict(model_probs)
        return self._explain_from_img(img_rgb, model_probs, vr)


def _verdict_to_3state(verdict: Verdict) -> str:
    if verdict in (Verdict.STRONG_FAKE, Verdict.LIKELY_FAKE):
        return "FAKE"
    if verdict in (Verdict.STRONG_REAL, Verdict.LIKELY_REAL):
        return "REAL"
    return "UNCERTAIN"
