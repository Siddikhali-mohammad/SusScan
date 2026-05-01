import os
os.environ.setdefault("OPENCV_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2; cv2.setNumThreads(1)
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.utils import set_seed, load_config
from src.inference.predict import DeepfakePredictor
from src.inference.visualize import (
    build_annotated, build_hotspot, img_to_b64,
)
from src.data.face_detector import (
    detect_and_crop_all, draw_face_boxes, normalize_face_orientation,
)

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
CONFIG_PATH    = os.getenv("CONFIG_PATH",    "configs/default.yaml")
ALLOWED_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 5-state verdicts that map to the legacy "FAKE"/"REAL" for visualize.py
_FAKE_VERDICTS = {"STRONG_FAKE", "LIKELY_FAKE"}
_REAL_VERDICTS = {"STRONG_REAL", "LIKELY_REAL"}


def _verdict_for_viz(prediction: str) -> str:
    if prediction in _FAKE_VERDICTS:
        return "FAKE"
    if prediction in _REAL_VERDICTS:
        return "REAL"
    return "UNCERTAIN"


predictor: DeepfakePredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg.seed)
    print(f"loading models from {CHECKPOINT_DIR} ...")
    predictor = DeepfakePredictor(CHECKPOINT_DIR, image_size=cfg.data.image_size)
    print("models ready.")
    yield
    predictor = None

app = FastAPI(title="Deepfake Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve frontend
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": predictor is not None}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 15 MB.")

    img = _decode_upload(data)
    if img.shape[0] < 32 or img.shape[1] < 32:
        raise HTTPException(status_code=400, detail="Image too small.")

    # detect all faces
    faces = detect_and_crop_all(img, padding=0.30)
    if not faces:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the image. Please upload a clear, front-facing photo."
        )

    all_bboxes = [b for _, b, _ in faces]

    # annotate original showing all detected faces
    img_annotated_full = draw_face_boxes(img, all_bboxes)

    # analyze every detected face
    face_results = []
    for face_idx, (raw_crop, bbox, lm) in enumerate(faces):
        face_crop, ori_status = normalize_face_orientation(raw_crop, lm)

        if ori_status["rejected"]:
            face_results.append({
                "face_index":      face_idx,
                "bbox":            list(bbox),
                "prediction":      "UNCERTAIN",
                "ui_color":        "amber",
                "ui_label":        "Cannot analyze",
                "ui_muted":        False,
                "reason":          ori_status["reason"],
                "triggered_rule":  "orientation_rejected",
                "confidence":      0.0,
                "ensemble_score":  0.5,
                "model_scores":    {"efb4": 0.0, "swin": 0.0, "clip": 0.0},
                "metrics":         {"spread": 0.0, "std": 0.0, "margin": 0.0,
                                    "votes_fake": 0, "votes_real": 0},
                "model_agreement": "split",
                "story": {
                    "headline":        "Cannot analyze — face orientation",
                    "summary":         ori_status["reason"],
                    "evidence":        [],
                    "model_take":      "Please upload an image where the face is upright.",
                    "model_breakdown": [],
                    "closing":         "",
                    "is_fake":         False,
                    "uncertain":       True,
                    "confidence":      0.0,
                    "agreement":       "split",
                    "reason":          ori_status["reason"],
                },
                "images": {
                    "face_crop": img_to_b64(cv2.resize(raw_crop, (224, 224))),
                    "annotated": img_to_b64(cv2.resize(raw_crop, (224, 224))),
                    "hotspot":   img_to_b64(cv2.resize(raw_crop, (224, 224))),
                },
            })
            continue

        # Pass the face crop array directly — no temp file, no lossy compression.
        try:
            result = predictor.predict_with_explanation_from_array(face_crop, use_tta=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed on face {face_idx + 1}: {e}")

        scores = result["individual_scores"]
        print(f"[face {face_idx}] efb4={scores['efb4']:.3f} swin={scores['swin']:.3f} "
              f"clip={scores['clip']:.3f} ensemble={result['ensemble_score']:.3f} "
              f"-> {result['prediction']} ({result['confidence']*100:.1f}%)")

        orig    = result["original_image"]
        pred    = result["prediction"]
        avg_cam = np.mean(list(result["heatmaps"].values()), axis=0)

        # Map to 3-state for visualize.py color selection
        viz_pred = _verdict_for_viz(pred)

        face_results.append({
            "face_index":      face_idx,
            "bbox":            list(bbox),          # ensure JSON-serializable
            "prediction":      result["prediction"],  # 5-state verdict string
            "ui_color":        result["ui_color"],
            "ui_label":        result["ui_label"],
            "ui_muted":        result["ui_muted"],
            "reason":          result["reason"],
            "triggered_rule":  result["triggered_rule"],
            "confidence":      round(result["confidence"] * 100, 1),
            "ensemble_score":  round(result["ensemble_score"], 4),
            "model_scores":    {k: round(v, 4) for k, v in result["individual_scores"].items()},
            "metrics": {
                "spread":      round(result["spread"], 4),
                "std":         round(result["std"], 4),
                "margin":      round(result["margin"], 4),
                "votes_fake":  result["votes_fake"],
                "votes_real":  result["votes_real"],
            },
            "model_agreement": result["model_agreement"],
            "story":           result["story"],
            "images": {
                "face_crop": img_to_b64(orig),
                "annotated": img_to_b64(build_annotated(orig, result["top_regions"], viz_pred)),
                "hotspot":   img_to_b64(build_hotspot(orig, avg_cam, viz_pred)),
            },
        })

    return {
        "faces_found":  len(faces),
        "input_full":   img_to_b64(img_annotated_full),
        "face_results": face_results,
    }


