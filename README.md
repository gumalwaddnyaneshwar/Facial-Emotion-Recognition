# DeepFER — Facial Emotion Recognition & Analytics

A full facial emotion recognition pipeline — from a retrained, evaluated
CNN through to a production-style application: multi-face detection,
real-time tracking, Grad-CAM explainability, satisfaction analytics, and
a documented REST API.

Built on MobileNetV2 (transfer learning), MediaPipe (face detection and
landmarks), and Streamlit/FastAPI for the application layer.

---

## Features

- **Multi-face emotion detection** — angry, disgust, fear, happy, neutral,
  sad, surprise, with full confidence breakdowns per face
- **Grad-CAM explainability** — visualizes which facial regions drove
  each prediction
- **Three input modes** — image upload (with batch support), video file
  analysis, and live webcam
- **Real-time multi-person tracking** — persistent per-person IDs with
  confirmation logic to reject false-positive detections, plus temporal
  smoothing to prevent frame-to-frame flicker
- **Satisfaction analytics dashboard** — session-level emotion timelines,
  a satisfaction score, notable-moment detection, CSV/PDF export
- **Session history** — SQLite-backed storage with cross-session trend
  comparison
- **Facial landmark overlay** and **age/gender estimation** (optional,
  toggleable)
- **Model A/B comparison** — the original vs. retrained model side by
  side, on the same input
- **REST API** (FastAPI, separate from the Streamlit app) with
  auto-generated Swagger documentation

---

## Project Structure

```
DeepFER_App/
├── app.py                  # Streamlit application (main entry point)
├── api.py                  # FastAPI REST service (independent of app.py)
├── face_detector.py        # MediaPipe-based face detection
├── emotion_inference.py    # Loads model, runs emotion classification
├── gradcam.py               # Grad-CAM explainability implementation
├── tracking.py             # CentroidTracker (multi-face ID) + EmotionSmoother
├── analytics.py            # Session-level stats, satisfaction scoring
├── history.py              # SQLite session storage + trend queries
├── report.py                # PDF report generation
├── landmarks.py            # MediaPipe FaceMesh overlay
├── age_gender.py           # Pretrained age/gender estimation (optional)
├── deepfer_mobilenetv2_v2.h5   # Retrained model weights
├── deepfer_mobilenetv2.h5      # Original (pre-retrain) model, for A/B comparison
├── class_labels.json       # Emotion class order
├── models/                  # Age/gender Caffe model files (see Setup)
├── .streamlit/
│   └── config.toml         # App theme
└── requirements.txt
```

---

## Setup

**1. Clone/copy the project and install dependencies:**
```bash
pip install -r requirements.txt
```
> Versions in `requirements.txt` are deliberately pinned — see the
> comments in that file for why. Installing "latest" for each package
> independently will reintroduce a real dependency conflict between
> TensorFlow, mediapipe, and jax.

**2. (Optional) Enable age & gender estimation.**
This feature uses a pretrained Caffe model, not one trained for this
project. Download these four files into a `models/` folder:

| File | Source |
|---|---|
| `age_deploy.prototxt` | [raw.githubusercontent.com/spmallick/learnopencv](https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt) |
| `gender_deploy.prototxt` | [raw.githubusercontent.com/spmallick/learnopencv](https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt) |
| `age_net.caffemodel` | [raw.githubusercontent.com/eveningglow](https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/age_net.caffemodel) |
| `gender_net.caffemodel` | [raw.githubusercontent.com/eveningglow](https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel) |

If these files aren't present, the app runs normally — the age/gender
toggle simply shows a warning and stays inactive.

**3. Run the app:**
```bash
streamlit run app.py
```

**4. (Optional) Run the API, independently of the app:**
```bash
uvicorn api:app --reload --port 8000
```
Then visit `http://localhost:8000/docs` for interactive documentation.

---

## Usage

- **Image upload** — upload one or more photos. Detects all faces, shows
  confidence breakdown, Grad-CAM heatmap, and (if enabled) landmarks,
  age/gender, and the model A/B comparison.
- **Video file** — upload a clip; samples frames at a configurable rate
  and produces a full session timeline and satisfaction summary.
- **Live webcam** — real-time multi-person tracking with on-screen labels;
  save completed sessions to History for later comparison.
- **History** — view, compare, or delete saved sessions; see satisfaction
  trends across multiple sessions.

Detection confidence, face crop padding, and webcam logging interval are
all configurable from the sidebar. **Note:** face crop padding above
~0.20 measurably degrades landmark and age/gender accuracy (see
Limitations) — keep it at 0.10–0.20 if those features are enabled.

---

## Model Training & Retraining

The emotion classifier is a MobileNetV2 backbone (ImageNet-pretrained)
with a custom classification head, trained on FER2013.

An initial version of the model produced weak, poorly-calibrated
predictions (25–55% confidence even on clear expressions) and
systematically suppressed the "surprise" class. Inspecting the saved
model directly (rather than guessing) revealed two causes:

1. **Label smoothing of 0.1** during training, which caps how confident
   the model is ever allowed to be
2. **Class imbalance** — "surprise" and "disgust" were underrepresented,
   and the model's output layer had learned a negative bias against them

**Fix:** retrained on Kaggle (P100 GPU) with label smoothing reduced to
0.05 and class-weighted training (`class_weight="balanced"`), plus light
data augmentation. Training ran 30 epochs with early stopping and
learning-rate decay.

| Metric | Before | After |
|---|---|---|
| Validation accuracy | ~42% (unstable) | **68.4%** |
| Typical top-class confidence | 25–55% | 60–95%+ |
| "Surprise" predictions | Suppressed | Precision 78.2%, Recall 81.6% |

68.4% is consistent with published human-level agreement on FER2013
(~65–68%), not an artificially inflated number.

---

## Evaluation Results

Full evaluation on the FER2013 test set (7,178 images):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Happy | 0.894 | 0.856 | 0.874 |
| Surprise | 0.782 | 0.816 | 0.799 |
| Disgust | 0.697 | 0.766 | 0.730 |
| Neutral | 0.613 | 0.668 | 0.639 |
| Angry | 0.597 | 0.610 | 0.603 |
| Sad | 0.575 | 0.554 | 0.565 |
| Fear | 0.549 | 0.517 | 0.532 |

**Overall accuracy: 68.4%**

The confusion matrix shows a genuine fear/sad/neutral/angry confusion
cluster — these four emotions visually overlap in static images, a
known, documented hard boundary in FER2013, not a defect specific to
this model.

---

## Known Limitations

- **Fear/sad/neutral/angry confusion** — these classes get mistaken for
  each other more than others (see Evaluation Results above).
- **High-energy "happy" expressions** (e.g. an open-mouth fist-pump) are
  sometimes misread as "surprise" — likely because FER2013's "happy"
  training examples skew toward calmer smiles rather than exaggerated
  celebratory expressions.
- **Age/gender estimation** uses an off-the-shelf pretrained model (not
  trained for this project) and is measurably sensitive to face crop
  padding (accuracy drops sharply above ~0.20) and to lighting
  conditions — verified through direct testing, not assumed.
- **Live webcam tracking** requires a few consecutive frames to confirm
  a new person before assigning a visible ID, by design — this trades a
  small delay for eliminating false-positive "phantom" IDs.

---

## Tech Stack

**ML / CV:** TensorFlow, Keras, MobileNetV2, MediaPipe, OpenCV
**App:** Streamlit
**API:** FastAPI, Uvicorn
**Data / storage:** pandas, SQLite
**Reporting:** ReportLab, Matplotlib
**Training:** Kaggle (P100 GPU)

---

## Acknowledgments

- FER2013 dataset for emotion classification training/evaluation
- Age/gender Caffe models from the [LearnOpenCV](https://github.com/spmallick/learnopencv)
  project (Levi & Hassner architecture)
- MediaPipe (Google) for face detection and face mesh landmarks
