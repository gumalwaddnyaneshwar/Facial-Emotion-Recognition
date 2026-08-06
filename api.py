"""
api.py — DeepFER REST API.

Exposes the face detection + emotion classification pipeline as a proper
HTTP service, independent of the Streamlit app. Useful for integrating
DeepFER into other applications, or as a standalone deployable service.

Run with:
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger documentation
(auto-generated from this file — every endpoint, parameter, and response
shape shown below is what appears there).
"""

import io
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from face_detector import FaceDetector
from emotion_inference import EmotionClassifier
from gradcam import GradCAM
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "deepfer_mobilenetv2_v2.h5"
LABELS_PATH = "class_labels.json"


# ─────────────────────────────────────────────────────────────
# Response schemas — these define exactly what /docs shows for
# each endpoint's response shape.
# ─────────────────────────────────────────────────────────────
class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FacePrediction(BaseModel):
    face_id: int
    box: FaceBox
    emotion: str
    confidence: float
    all_scores: dict


class PredictResponse(BaseModel):
    faces_detected: int
    predictions: List[FacePrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str]


# ─────────────────────────────────────────────────────────────
# App setup — models load once at startup, not per-request
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepFER API",
    description=(
        "Facial emotion recognition service. Upload an image to detect "
        "faces and classify each one's emotion (angry, disgust, fear, "
        "happy, neutral, sad, surprise), with full confidence breakdowns "
        "and optional Grad-CAM explainability heatmaps."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to specific domains before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[FaceDetector] = None
classifier: Optional[EmotionClassifier] = None
cam: Optional[GradCAM] = None


@app.on_event("startup")
def load_models():
    """Load the face detector, classifier, and Grad-CAM helper once when
    the server starts, so requests don't pay model-loading cost each time.
    """
    global detector, classifier, cam
    detector = FaceDetector()
    classifier = EmotionClassifier(MODEL_PATH, LABELS_PATH)
    cam = GradCAM(classifier.model)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def read_upload_as_bgr(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR array."""
    try:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """Basic service info — points to the interactive docs."""
    return {
        "service": "DeepFER API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/predict/gradcam"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Health check — confirms the service is up and the model loaded correctly."""
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        classes=classifier.class_labels if classifier else [],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Image file (jpg/png)")):
    """Detect all faces in an uploaded image and classify each one's emotion.

    Returns, for every detected face: its bounding box, the top predicted
    emotion, that prediction's confidence, and the full probability
    breakdown across all 7 classes.
    """
    if classifier is None or detector is None:
        raise HTTPException(status_code=503, detail="Models are still loading — try again shortly.")

    frame_bgr = read_upload_as_bgr(await file.read())
    faces = detector.detect(frame_bgr)

    predictions = []
    for i, face in enumerate(faces):
        crop = detector.crop(frame_bgr, face, target_size=(224, 224))
        if crop is None:
            continue
        result = classifier.predict(crop)
        x, y, w, h = face["box"]
        predictions.append(FacePrediction(
            face_id=i,
            box=FaceBox(x=x, y=y, width=w, height=h),
            emotion=result["emotion"],
            confidence=result["confidence"],
            all_scores=result["all_scores"],
        ))

    return PredictResponse(faces_detected=len(predictions), predictions=predictions)


@app.post("/predict/gradcam", tags=["Prediction"])
async def predict_gradcam(
    file: UploadFile = File(..., description="Image file (jpg/png)"),
    face_index: int = Query(0, description="Which detected face to explain, if multiple (0 = first)"),
):
    """Return a Grad-CAM heatmap overlay (as a PNG image) showing which
    facial regions drove the prediction for one face in the image.

    If multiple faces are detected, use `face_index` to pick which one —
    defaults to the first face found. Response is a raw PNG image, so it
    can be displayed directly (e.g. `<img src="...">`) or saved to disk.
    """
    if classifier is None or detector is None or cam is None:
        raise HTTPException(status_code=503, detail="Models are still loading — try again shortly.")

    frame_bgr = read_upload_as_bgr(await file.read())
    faces = detector.detect(frame_bgr)

    if not faces:
        raise HTTPException(status_code=404, detail="No faces detected in the uploaded image.")
    if face_index < 0 or face_index >= len(faces):
        raise HTTPException(
            status_code=400,
            detail=f"face_index {face_index} out of range — {len(faces)} face(s) detected (valid: 0-{len(faces) - 1}).",
        )

    crop = detector.crop(frame_bgr, faces[face_index], target_size=(224, 224))
    if crop is None:
        raise HTTPException(status_code=422, detail="Could not crop the selected face (too close to image edge).")

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    batch = preprocess_input(np.expand_dims(crop_rgb.astype(np.float32), axis=0))

    result = classifier.predict(crop)
    class_index = classifier.class_labels.index(result["emotion"])
    heatmap, _ = cam.compute(batch, class_index=class_index)
    overlay_bgr = cam.overlay_on_image(crop, heatmap)

    success, encoded = cv2.imencode(".png", overlay_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode heatmap image.")

    return Response(content=encoded.tobytes(), media_type="image/png")
