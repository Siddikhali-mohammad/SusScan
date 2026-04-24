import numpy as np

# Each entry: (key, label, row_start, row_end, col_start, col_end) — fractions of 224×224
# Face is assumed to be centered and filling the frame (pre-cropped inputs)
_REGIONS = [
    ("forehead",    "Forehead",               0.02, 0.28, 0.18, 0.82),
    ("eyes",        "Eye Area",               0.28, 0.46, 0.08, 0.92),
    ("nose",        "Nose Bridge",            0.44, 0.62, 0.28, 0.72),
    ("mouth",       "Mouth / Lips",           0.60, 0.76, 0.22, 0.78),
    ("chin",        "Chin / Jaw",             0.74, 0.92, 0.18, 0.82),
    ("left_edge",   "Left Blending Edge",     0.20, 0.82, 0.00, 0.18),
    ("right_edge",  "Right Blending Edge",    0.20, 0.82, 0.82, 1.00),
]


def map_regions(cam, top_n=3):
    H, W = cam.shape
    results = []
    for key, label, r0, r1, c0, c1 in _REGIONS:
        patch = cam[int(r0*H):int(r1*H), int(c0*W):int(c1*W)]
        results.append({
            "region": key,
            "label": label,
            "intensity": float(patch.mean()),
        })
    results.sort(key=lambda x: x["intensity"], reverse=True)
    return results[:top_n]
