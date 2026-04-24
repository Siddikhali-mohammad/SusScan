import base64
import cv2
import numpy as np

from ..explainability.region_mapping import _REGIONS

_REGION_COORDS = {key: (r0, r1, c0, c1) for key, _, r0, r1, c0, c1 in _REGIONS}

_FAKE_COLORS      = [(220, 60,  60),  (220, 140, 60),  (220, 200, 60)]
_REAL_COLORS      = [(60,  200, 80),  (60,  200, 160), (60,  160, 220)]
_UNCERTAIN_COLORS = [(210, 160, 40),  (210, 130, 40),  (200, 100, 40)]

_ARTIFACT_SHORT = {
    "eyes":        "eye blending",
    "left_edge":   "left edge artifact",
    "right_edge":  "right edge artifact",
    "chin":        "jawline seam",
    "mouth":       "mouth artifact",
    "forehead":    "forehead texture",
    "nose":        "nose warp",
}


def _pred_colors(pred):
    if pred == "FAKE":      return _FAKE_COLORS
    if pred == "REAL":      return _REAL_COLORS
    return _UNCERTAIN_COLORS


def build_annotated(orig, top_regions, pred):
    colors = _pred_colors(pred)
    H, W = orig.shape[:2]

    overlay = orig.copy()
    for i, r in enumerate(top_regions[:3]):
        r0, r1, c0, c1 = _REGION_COORDS[r["region"]]
        y1, y2 = int(r0 * H), int(r1 * H)
        x1, x2 = int(c0 * W), int(c1 * W)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colors[i], -1)

    out = cv2.addWeighted(overlay, 0.35, orig, 0.65, 0)

    for i, r in enumerate(top_regions[:3]):
        r0, r1, c0, c1 = _REGION_COORDS[r["region"]]
        y1, y2 = int(r0 * H), int(r1 * H)
        x1, x2 = int(c0 * W), int(c1 * W)
        cv2.rectangle(out, (x1, y1), (x2, y2), colors[i], 2)
        label = _ARTIFACT_SHORT.get(r["region"], r["label"])
        cv2.putText(out, f"{i+1}. {label}", (x1 + 3, y1 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, colors[i], 1, cv2.LINE_AA)
    return out


def build_hotspot(orig, avg_cam, pred):
    if pred == "FAKE":      color = (220, 60, 60)
    elif pred == "REAL":    color = (60, 200, 80)
    else:                   color = (210, 160, 40)
    thresh = np.percentile(avg_cam, 70)
    mask = (avg_cam > thresh).astype(np.uint8)

    tinted = np.zeros_like(orig)
    tinted[mask == 1] = color
    out = cv2.addWeighted(orig, 0.55, tinted, 0.45, 0)

    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def build_verdict_panel(result, H, W):
    pred   = result["prediction"]
    conf   = result["confidence"]
    scores = result["individual_scores"]
    top    = result["top_regions"]

    panel         = np.full((H, W, 3), 25, dtype=np.uint8)
    if pred == "FAKE":   verdict_color = (80, 80, 230)
    elif pred == "REAL": verdict_color = (80, 210, 80)
    else:                verdict_color = (210, 160, 40)
    accent  = (200, 200, 200)
    dim     = (130, 130, 130)
    colors  = _pred_colors(pred)

    def txt(s, x, y, scale, color, thick=1):
        cv2.putText(panel, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    y = 22
    txt(pred, 8, y, 0.80, verdict_color, 2);  y += 22
    txt(f"{conf*100:.1f}% confident", 8, y, 0.38, accent);  y += 18
    cv2.line(panel, (6, y), (W - 6, y), (55, 55, 55), 1);  y += 10

    txt("Model scores:", 8, y, 0.34, dim);  y += 14
    for name, score in scores.items():
        bar_len   = int(score * (W - 60))
        bar_color = (80, 80, 200) if score >= 0.5 else (80, 180, 80)
        cv2.rectangle(panel, (8, y - 8), (8 + bar_len, y), bar_color, -1)
        txt(f"{name} {score:.2f}", 10, y - 1, 0.28, (240, 240, 240))
        y += 13

    cv2.line(panel, (6, y), (W - 6, y), (55, 55, 55), 1);  y += 9
    txt("Top signals:", 8, y, 0.34, dim);  y += 14
    for i, r in enumerate(top[:3]):
        label = _ARTIFACT_SHORT.get(r["region"], r["label"])
        txt(f"{i+1}. {label}", 10, y, 0.30, colors[i])
        y += 13

    return panel


def img_to_b64(img_rgb, quality=92):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def build_combined_png(result):
    orig    = result["original_image"]
    H, W    = orig.shape[:2]
    pred    = result["prediction"]
    avg_cam = np.mean(list(result["heatmaps"].values()), axis=0)

    panels = [
        (orig,                                                   "Original"),
        (build_annotated(orig, result["top_regions"], pred),     "Detected Regions"),
        (build_hotspot(orig, avg_cam, pred),                     "Activation Hotspot"),
        (build_verdict_panel(result, H, W),                      "Verdict"),
    ]

    label_h = 18
    canvas  = np.full((H + label_h, W * 4, 3), 15, dtype=np.uint8)

    for i, (panel, title) in enumerate(panels):
        x = i * W
        canvas[label_h:, x:x + W] = panel
        cv2.putText(canvas, title, (x + 4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        if i > 0:
            cv2.line(canvas, (x, 0), (x, H + label_h), (60, 60, 60), 1)

    _, buf = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return bytes(buf)
