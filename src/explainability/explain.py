_MODEL_NAMES = {
    "efb4": "EfficientNet-B4",
    "swin": "Swin Transformer",
    "clip": "CLIP ViT-B/16",
}

_REGION_HINTS = {
    "eyes":       "eye-region blending inconsistencies, common in face-swap deepfakes",
    "left_edge":  "boundary compositing artifacts along the left face edge",
    "right_edge": "boundary compositing artifacts along the right face edge",
    "chin":       "jawline blending seams, typical of face-swap and reenactment fakes",
    "mouth":      "mouth-region artifacts, often seen in expression-transfer manipulations",
    "forehead":   "forehead texture inconsistencies or hairline boundary artifacts",
    "nose":       "nose-bridge synthesis artifacts from 3D warping",
}

_REGION_DESCRIPTION = {
    "eyes":       "The eye area shows blending inconsistencies where the fake face was composited onto the source.",
    "left_edge":  "The left edge of the face shows compositing artifacts where the face boundary was blended in.",
    "right_edge": "The right edge of the face shows compositing artifacts where the face boundary was blended in.",
    "chin":       "The jawline shows blending seams characteristic of face-swap or reenactment manipulation.",
    "mouth":      "The mouth region contains synthesis artifacts typical of expression-transfer deepfakes.",
    "forehead":   "The forehead shows texture inconsistencies or hairline boundary artifacts from the swap.",
    "nose":       "The nose bridge contains synthesis artifacts likely introduced by 3D warping during generation.",
}

# 5-state verdicts that map to fake/real/uncertain
_FAKE_VERDICTS      = {"STRONG_FAKE", "LIKELY_FAKE", "FAKE"}
_REAL_VERDICTS      = {"STRONG_REAL", "LIKELY_REAL", "REAL"}
_UNCERTAIN_VERDICTS = {"UNCERTAIN"}


def model_agreement(scores, prediction):
    if prediction in _FAKE_VERDICTS:
        votes = sum(1 for s in scores.values() if s >= 0.5)
    elif prediction in _REAL_VERDICTS:
        votes = sum(1 for s in scores.values() if s < 0.5)
    else:  # UNCERTAIN — models are inherently disagreeing
        return "split"

    if votes == 3: return "unanimous"
    if votes == 2: return "majority"
    return "split"


def build_story(prediction, confidence, ensemble_scores, top_regions, agreement, reason=None):
    conf_pct  = round(confidence * 100, 1)
    is_fake   = prediction in _FAKE_VERDICTS
    uncertain = prediction in _UNCERTAIN_VERDICTS

    # headline
    if prediction == "STRONG_REAL":
        headline = f"AUTHENTIC — {conf_pct}% confident"
    elif prediction == "LIKELY_REAL":
        headline = f"PROBABLY AUTHENTIC — {conf_pct}% confident"
    elif prediction in ("UNCERTAIN", "REAL", "FAKE"):
        # legacy 3-state or genuine uncertain
        if uncertain:
            headline = f"UNCERTAIN — {conf_pct}% fake probability (inconclusive)"
        else:
            headline = f"{'MANIPULATED' if is_fake else 'AUTHENTIC'} — {conf_pct}% confident"
    elif prediction == "LIKELY_FAKE":
        headline = f"PROBABLY MANIPULATED — {conf_pct}% confident"
    elif prediction == "STRONG_FAKE":
        headline = f"MANIPULATED — {conf_pct}% confident"
    else:
        headline = f"{'MANIPULATED' if is_fake else 'AUTHENTIC'} — {conf_pct}% confident"

    # one-line summary
    if prediction == "STRONG_REAL":
        summary = "All three models agree this image is genuine."
    elif prediction == "LIKELY_REAL":
        summary = "Most models indicate this is authentic, though one disagrees."
    elif prediction == "LIKELY_FAKE":
        summary = (
            "Most models indicate manipulation, though one disagrees. "
            "Portrait mode or AI-enhanced photos can also trigger this result."
        )
    elif prediction == "STRONG_FAKE":
        summary = "All three models flagged this image as manipulated."
    elif uncertain:
        summary = (
            "The ensemble score is in the uncertain zone (40–62%). "
            "The models cannot confidently determine whether this image is authentic or manipulated. "
            "Manual review is recommended."
        )
    elif is_fake:
        summary = (
            "This image contains signs of AI-generated or digitally manipulated facial content. "
            "Note: portrait mode photos and AI-enhanced images can also trigger this verdict "
            "due to similar edge compositing artifacts."
        )
    else:
        summary = "This image appears to be a genuine, unmodified photograph."

    evidence = []
    for r in top_regions:
        evidence.append({
            "region":      r["region"],
            "label":       r["label"],
            "intensity":   round(r["intensity"], 4),
            "description": _REGION_DESCRIPTION.get(r["region"], f"Activation detected in the {r['label']} region."),
            "hint":        _REGION_HINTS.get(r["region"], ""),
        })

    if uncertain:
        model_take = (
            "The models disagree or are individually uncertain — the ensemble score sits in the "
            "undecided range. This result should not be acted on without additional context."
        )
    elif agreement == "unanimous":
        model_take = f"All three models ({', '.join(_MODEL_NAMES.values())}) independently agree on this verdict."
    elif agreement == "majority":
        if is_fake:
            agreeing = [_MODEL_NAMES[k] for k, v in ensemble_scores.items() if v >= 0.5]
        else:
            agreeing = [_MODEL_NAMES[k] for k, v in ensemble_scores.items() if v < 0.5]
        model_take = f"{' and '.join(agreeing)} flag this as {'manipulated' if is_fake else 'authentic'}, with one model uncertain."
    else:
        model_take = "The models are split — treat this result with caution. Manual review is recommended."

    model_breakdown = [
        {"key": k, "name": _MODEL_NAMES[k], "score": round(v, 4)}
        for k, v in ensemble_scores.items()
    ]

    if uncertain:
        scores_str = ", ".join(f"{_MODEL_NAMES[k]} {v*100:.1f}%" for k, v in ensemble_scores.items())
        closing = f"Individual model scores: {scores_str}."
    elif is_fake:
        best  = max(ensemble_scores, key=ensemble_scores.get)
        worst = min(ensemble_scores, key=ensemble_scores.get)
        closing = (
            f"{_MODEL_NAMES[best]} was most certain at {ensemble_scores[best]*100:.1f}%, "
            f"while {_MODEL_NAMES[worst]} was least confident at {ensemble_scores[worst]*100:.1f}%."
        )
        if agreement == "majority":
            closing += (
                " Only 2 of 3 models agree — if this is a portrait mode or AI-enhanced photo, "
                "that can produce similar boundary artifacts to a face swap."
            )
    else:
        lowest = min(ensemble_scores, key=ensemble_scores.get)
        if ensemble_scores[lowest] > 0.3:
            closing = (
                f"{_MODEL_NAMES[lowest]} showed slightly elevated fake probability "
                f"({ensemble_scores[lowest]*100:.1f}%) — worth a second look if the stakes are high."
            )
        else:
            closing = "All models show low fake probability across the board."

    return {
        "headline":        headline,
        "summary":         summary,
        "evidence":        evidence,
        "model_take":      model_take,
        "model_breakdown": model_breakdown,
        "closing":         closing,
        "is_fake":         is_fake,
        "uncertain":       uncertain,
        "confidence":      conf_pct,
        "agreement":       agreement,
        "reason":          reason or "",
    }


def generate_explanation(prediction, confidence, ensemble_scores, top_regions, agreement):
    # returns a single string rather than the structured dict — used by CLI
    story = build_story(prediction, confidence, ensemble_scores, top_regions, agreement)
    lines = [story["summary"]]

    if story["agreement"] == "unanimous":
        lines.append("All three models agree on this assessment.")
    elif story["agreement"] == "majority":
        lines.append(story["model_take"])
    else:
        lines.append(story["model_take"])

    if story["evidence"] and story["is_fake"]:
        top2 = [e["label"] for e in story["evidence"][:2]]
        lines.append(
            f"Strongest signals in: {' and '.join(top2)}. "
            f"This pattern is consistent with {story['evidence'][0]['hint']}."
        )

    lines.append(story["closing"])
    return " ".join(lines)
