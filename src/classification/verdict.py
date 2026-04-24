from dataclasses import dataclass, field
from enum import Enum
import statistics


class Verdict(str, Enum):
    STRONG_REAL  = "STRONG_REAL"
    LIKELY_REAL  = "LIKELY_REAL"
    UNCERTAIN    = "UNCERTAIN"
    LIKELY_FAKE  = "LIKELY_FAKE"
    STRONG_FAKE  = "STRONG_FAKE"


# UI rendering metadata per verdict
_VERDICT_META = {
    Verdict.STRONG_REAL:  {"ui_color": "green",  "ui_label": "Likely authentic",     "muted": False},
    Verdict.LIKELY_REAL:  {"ui_color": "green",  "ui_label": "Probably authentic",   "muted": True},
    Verdict.UNCERTAIN:    {"ui_color": "amber",  "ui_label": "Inconclusive",         "muted": False},
    Verdict.LIKELY_FAKE:  {"ui_color": "red",    "ui_label": "Probably manipulated", "muted": True},
    Verdict.STRONG_FAKE:  {"ui_color": "red",    "ui_label": "Likely manipulated",   "muted": False},
}


@dataclass
class VerdictResult:
    verdict:       Verdict
    ui_color:      str
    ui_label:      str
    ui_muted:      bool
    reason:        str
    triggered_rule: str
    base_probs:    dict
    ensemble_prob: float
    spread:        float
    std:           float
    margin:        float
    votes_fake:    int
    votes_real:    int


def classify(
    efb4_prob: float,
    swin_prob: float,
    clip_prob: float,
    ensemble_prob: float,
) -> VerdictResult:
    probs      = [efb4_prob, swin_prob, clip_prob]
    spread     = max(probs) - min(probs)
    std        = statistics.stdev(probs)
    margin     = abs(ensemble_prob - 0.5)
    votes_fake = sum(1 for p in probs if p > 0.5)
    votes_real = sum(1 for p in probs if p < 0.5)

    base_probs = {"efb4": efb4_prob, "swin": swin_prob, "clip": clip_prob}

    def make(verdict, reason, rule):
        meta = _VERDICT_META[verdict]
        return VerdictResult(
            verdict=verdict, ui_color=meta["ui_color"], ui_label=meta["ui_label"],
            ui_muted=meta["muted"], reason=reason, triggered_rule=rule,
            base_probs=base_probs, ensemble_prob=ensemble_prob,
            spread=spread, std=std, margin=margin,
            votes_fake=votes_fake, votes_real=votes_real,
        )

    # Rule 1: high spread → UNCERTAIN, unless 2-of-3 majority already agrees strongly.
    # "convincingly real" means max real voter < 0.45, not just barely below 0.5 —
    # otherwise portrait-mode borderline cases slip through the abstain.
    real_voters    = [p for p in probs if p < 0.5]
    _majority_fake = votes_fake >= 2 and ensemble_prob > 0.65
    _majority_real = (votes_real >= 2 and ensemble_prob < 0.35
                      and (max(real_voters) < 0.45 if real_voters else False))
    if spread > 0.25 and not (_majority_fake or _majority_real):
        names = ["efb4", "swin", "clip"]
        high  = names[probs.index(max(probs))]
        low   = names[probs.index(min(probs))]
        return make(
            Verdict.UNCERTAIN,
            f"Models disagree too much (spread={spread:.2f}): {high} leans fake, {low} leans real. Cannot reach a confident verdict.",
            "rule_1_spread",
        )

    # Rule 2: ensemble too close to the boundary
    if margin < 0.10:
        return make(
            Verdict.UNCERTAIN,
            f"Ensemble score {ensemble_prob:.2f} is too close to the 0.5 boundary (margin={margin:.2f}). This image likely has mixed signals.",
            "rule_2_margin",
        )

    # Rule 3: all three say real, tight spread
    if votes_real == 3 and ensemble_prob < 0.35 and spread < 0.15:
        return make(
            Verdict.STRONG_REAL,
            "All three models agree this image looks authentic.",
            "rule_3_strong_real",
        )

    # Rule 4: all three say fake, tight spread
    if votes_fake == 3 and ensemble_prob > 0.65 and spread < 0.15:
        return make(
            Verdict.STRONG_FAKE,
            "All three models agree this image appears manipulated.",
            "rule_4_strong_fake",
        )

    # Rule 5: 2-of-3 say real
    if votes_real >= 2 and ensemble_prob < 0.45:
        dissenters = [n for n, p in zip(["efb4", "swin", "clip"], probs) if p > 0.5]
        d = dissenters[0] if dissenters else "one model"
        return make(
            Verdict.LIKELY_REAL,
            f"Two of three models indicate authentic; {d} disagrees.",
            "rule_5_likely_real",
        )

    # Rule 6: 2-of-3 say fake
    if votes_fake >= 2 and ensemble_prob > 0.55:
        dissenters = [n for n, p in zip(["efb4", "swin", "clip"], probs) if p < 0.5]
        d = dissenters[0] if dissenters else "one model"
        return make(
            Verdict.LIKELY_FAKE,
            f"Two of three models indicate manipulation; {d} disagrees.",
            "rule_6_likely_fake",
        )

    # Rule 7: nothing matched
    return make(
        Verdict.UNCERTAIN,
        "Signals are mixed and don't meet the threshold for a confident verdict.",
        "rule_7_fallback",
    )
