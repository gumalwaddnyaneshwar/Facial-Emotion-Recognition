"""
app.py — DeepFER: Customer Satisfaction & Emotion Analytics Tool.

Three modes:
  - Image upload: single photo, full breakdown + Grad-CAM heatmap per face.
  - Video file: analyze a pre-recorded clip, emotion timeline + summary.
  - Live webcam: real-time multi-face tracking with a live analytics feed.

Run with:
    streamlit run app.py
"""

import time
import tempfile

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from face_detector import FaceDetector
from emotion_inference import EmotionClassifier
from gradcam import GradCAM
from analytics import EmotionSession
from tracking import CentroidTracker, EmotionSmoother
from report import generate_pdf_report
from landmarks import LandmarkOverlay
from age_gender import AgeGenderEstimator
import history
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "deepfer_mobilenetv2_v2.h5"
OLD_MODEL_PATH = "deepfer_mobilenetv2.h5"  # pre-retraining model, for the A/B comparison view
LABELS_PATH = "class_labels.json"
AGE_GENDER_MODEL_DIR = "models"  # see age_gender.py docstring for what goes here

EMOTION_COLORS = {  # BGR, for on-frame box/label drawing
    "happy": (0, 200, 0),
    "surprise": (0, 200, 255),
    "neutral": (200, 200, 200),
    "sad": (255, 120, 0),
    "angry": (0, 0, 255),
    "fear": (200, 0, 200),
    "disgust": (0, 128, 128),
}

EMOTION_EMOJI = {
    "happy": "😄", "surprise": "😲", "neutral": "😐",
    "sad": "😢", "angry": "😠", "fear": "😨", "disgust": "🤢",
}

EMOTION_HEX = {  # for CSS badges/charts — matches EMOTION_COLORS but in RGB hex
    "happy": "#22C55E", "surprise": "#38BDF8", "neutral": "#94A3B8",
    "sad": "#3B82F6", "angry": "#EF4444", "fear": "#C026D3", "disgust": "#14B8A6",
}


# ─────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector(min_confidence=0.5, padding_ratio=0.25):
    return FaceDetector(min_confidence=min_confidence, padding_ratio=padding_ratio)


@st.cache_resource
def load_classifier():
    return EmotionClassifier(MODEL_PATH, LABELS_PATH)


@st.cache_resource
def load_old_classifier():
    """The pre-retraining model, loaded only when the A/B comparison view
    is used — kept separate from the main classifier so the app doesn't
    pay the extra load cost unless someone actually wants the comparison.
    """
    return EmotionClassifier(OLD_MODEL_PATH, LABELS_PATH)


@st.cache_resource
def load_gradcam(_classifier):
    return GradCAM(_classifier.model)


@st.cache_resource
def load_landmark_overlay():
    return LandmarkOverlay()


@st.cache_resource
def load_age_gender():
    """Returns None (instead of raising) if the model files aren't
    downloaded yet — callers check for None and show a helpful message
    rather than crashing the whole app over an optional feature.
    """
    try:
        return AgeGenderEstimator(model_dir=AGE_GENDER_MODEL_DIR)
    except FileNotFoundError:
        return None


def classify_face(classifier, crop_bgr):
    """Run the classifier and return (emotion, confidence, all_scores)."""
    result = classifier.predict(crop_bgr)
    return result["emotion"], result["confidence"], result["all_scores"]


def render_emotion_badge(emotion: str, confidence: float):
    """Render a colored pill-shaped badge for an emotion, matching the
    emotion's assigned color — used in place of plain st.metric text so
    the predicted emotion is immediately visually distinct.
    """
    color = EMOTION_HEX.get(emotion, "#94A3B8")
    emoji = EMOTION_EMOJI.get(emotion, "")
    st.markdown(
        f"""
        <div style="
            display:inline-flex; align-items:center; gap:8px;
            background:{color}22; border:1px solid {color};
            border-radius:999px; padding:10px 20px; margin-bottom:8px;
        ">
            <span style="font-size:1.4rem;">{emoji}</span>
            <span style="font-size:1.15rem; font-weight:600; color:{color};">
                {emotion.capitalize()}
            </span>
            <span style="font-size:0.9rem; color:#94A3B8;">
                {confidence * 100:.1f}% confidence
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_label(frame, box, emotion, confidence, label_prefix=None):
    """Draw a bounding box + emotion label on a frame (multi-face safe).

    Args:
        emotion: the raw emotion class (used for color lookup — keep this
            the actual label, e.g. "happy", not a combined display string).
        label_prefix: optional text shown before the emotion, e.g. "Person 0"
            — kept separate from `emotion` so color lookup still works.
    """
    x, y, w, h = box
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label_prefix}: {emotion} {confidence * 100:.0f}%" if label_prefix else \
           f"{emotion} {confidence * 100:.0f}%"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def render_dashboard(session: EmotionSession, key_prefix: str):
    """Shared analytics dashboard — used by both video and webcam modes."""
    summary = session.summary()

    if summary["total_predictions"] == 0:
        st.info("No data yet.")
        return

    st.markdown("### 📊 Session Analytics")

    if summary["total_predictions"] < 5:
        st.caption(
            f"⚠️ Only {summary['total_predictions']} prediction(s) in this session — "
            f"'Dominant emotion' and 'Satisfaction score' below are based on very few "
            f"samples and may not represent the full clip reliably."
        )

    st.markdown('<div class="deepfer-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total predictions", summary["total_predictions"])
    dominant = summary["dominant_emotion"]
    col2.metric("Dominant emotion", f"{EMOTION_EMOJI.get(dominant, '')} {dominant.capitalize()}")
    score = summary["satisfaction_score"]
    col3.metric("Satisfaction score", f"{score:+.1f}",
                help="% Satisfied minus % Dissatisfied. Range: -100 to +100.")
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Emotion distribution**")
        dist_df = pd.DataFrame(
            list(summary["emotion_distribution"].items()), columns=["Emotion", "Percent"]
        ).set_index("Emotion")
        st.bar_chart(dist_df)

    with col_b:
        st.markdown("**Satisfaction breakdown**")
        sat_df = pd.DataFrame(
            list(summary["satisfaction_distribution"].items()), columns=["Category", "Percent"]
        ).set_index("Category")
        st.bar_chart(sat_df)

    st.markdown("**Emotion over time**")
    df = session.to_dataframe()
    timeline = df.pivot_table(index="timestamp", columns="emotion", values="confidence", aggfunc="mean")
    st.line_chart(timeline)

    moments = session.notable_moments()
    if moments:
        st.markdown("**Notable moments** (high-confidence, non-neutral)")
        for m in moments[:10]:
            st.write(f"- {m['timestamp']:.1f}s — Person {m['face_id']}: "
                      f"**{m['emotion']}** ({m['confidence'] * 100:.0f}%)")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        st.download_button(
            "⬇️ Download CSV",
            data=session.to_csv_bytes(),
            file_name="deepfer_session.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_csv",
        )

    with export_col2:
        st.download_button(
            "📄 Download PDF report",
            data=generate_pdf_report(session),
            file_name="deepfer_session_report.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_download_pdf",
        )

    with export_col3:
        if st.button("💾 Save to history", key=f"{key_prefix}_save_history"):
            history.save_session(session, source=key_prefix)
            st.success("Session saved to history.")


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFER — Emotion Analytics",
    page_icon="🎭",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Tighten default top padding so the hero banner sits closer to the top */
    .block-container { padding-top: 1.5rem; }

    /* Hero header */
    .deepfer-hero {
        background: linear-gradient(135deg, #6C5CE7 0%, #341f97 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .deepfer-hero h1 {
        color: white; font-size: 1.9rem; margin: 0 0 6px 0; font-weight: 700;
    }
    .deepfer-hero p {
        color: #E0DEFB; font-size: 1rem; margin: 0;
    }

    /* Section cards used around dashboard content */
    .deepfer-card {
        background: #1A1D27; border: 1px solid #2A2E3D;
        border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;
    }

    /* Sidebar branding */
    .deepfer-sidebar-brand {
        font-size: 1.1rem; font-weight: 700; color: #E8E8F0;
        padding: 4px 0 12px 0; border-bottom: 1px solid #2A2E3D; margin-bottom: 12px;
    }
    </style>

    <div class="deepfer-hero">
        <h1>🎭 DeepFER — Facial Emotion Recognition</h1>
        <p>Multi-face detection · Grad-CAM explainability · Satisfaction analytics · Real-time tracking</p>
    </div>
    """,
    unsafe_allow_html=True,
)

history.init_db()

st.sidebar.markdown('<div class="deepfer-sidebar-brand">🎭 DeepFER</div>', unsafe_allow_html=True)

with st.sidebar.expander("⚙️ Detection settings"):
    min_confidence = st.slider(
        "Face detection confidence", 0.3, 0.9, 0.5, 0.05,
        help="Higher = fewer false-positive face detections, but may miss angled/partial faces.",
    )
    padding_ratio = st.slider(
        "Face crop padding", 0.0, 0.5, 0.15, 0.05,
        help="Extra margin around each detected face. Keep this at 0.20 or below if you're using "
             "facial landmarks or age/gender — both degrade noticeably above that, since the face "
             "ends up occupying too little of the crop for those models to work well.",
    )
    webcam_log_interval = st.slider(
        "Webcam logging interval (sec)", 0.5, 3.0, 1.0, 0.5,
        help="How often to log a prediction per person during a live webcam session.",
    )

with st.sidebar.expander("🧪 Advanced features"):
    show_landmarks = st.checkbox(
        "Show facial landmarks", value=False,
        help="Overlay the 468-point face mesh — purely visual, doesn't affect predictions.",
    )
    show_age_gender = st.checkbox(
        "Estimate age & gender", value=False,
        help="Requires the pretrained Caffe model files — see age_gender.py for download links "
             "if this shows a 'model files not found' warning.",
    )
    show_ab_compare = st.checkbox(
        "Compare with original (pre-retrain) model", value=False,
        help="Image upload mode only — shows both models' predictions side by side.",
    )

if padding_ratio > 0.20 and (show_landmarks or show_age_gender):
    st.sidebar.warning(
        f"Padding is {padding_ratio:.2f} — landmarks and age/gender accuracy drop sharply above "
        "0.20, since the face becomes too small within the crop for those models. "
        "Consider lowering it to 0.10-0.20."
    )

with st.spinner("Loading models..."):
    detector = load_detector(min_confidence=min_confidence, padding_ratio=padding_ratio)
    classifier = load_classifier()
    cam = load_gradcam(classifier)
    landmark_overlay = load_landmark_overlay() if show_landmarks else None
    age_gender_estimator = load_age_gender() if show_age_gender else None
    old_classifier = load_old_classifier() if show_ab_compare else None

    if show_age_gender and age_gender_estimator is None:
        st.sidebar.warning(
            "Age/gender model files not found in `models/` — see age_gender.py "
            "for download links. Feature disabled until files are added."
        )

mode = st.sidebar.radio(
    "Mode",
    ["📷  Image upload", "🎥  Video file", "📹  Live webcam", "📈  History"],
    label_visibility="collapsed",
)
mode = mode.split("  ", 1)[1]  # strip the icon prefix so existing mode-matching logic below is untouched


# ─────────────────────────────────────────────────────────────
# MODE 1 — Image upload (multi-face, confidence breakdown, Grad-CAM)
# ─────────────────────────────────────────────────────────────
def process_uploaded_image(pil_image):
    """Run detection + classification + Grad-CAM on one image and render
    the results — shared by both single and batch image uploads.
    """
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    st.image(pil_image, caption="Uploaded image", use_container_width=True)

    faces = detector.detect(frame_bgr)

    if not faces:
        st.warning("No faces detected. Try a clearer, front-facing photo.")
        return

    st.success(f"Found {len(faces)} face(s).")

    for i, face in enumerate(faces):
        st.markdown('<div class="deepfer-card">', unsafe_allow_html=True)
        st.markdown(f"#### Face {i + 1}")
        crop = detector.crop(frame_bgr, face, target_size=(224, 224))
        if crop is None:
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        emotion, confidence, all_scores = classify_face(classifier, crop)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        batch = preprocess_input(np.expand_dims(crop_rgb.astype(np.float32), axis=0))
        class_index = classifier.class_labels.index(emotion)
        heatmap, _ = cam.compute(batch, class_index=class_index)
        overlay_rgb = cv2.cvtColor(cam.overlay_on_image(crop, heatmap), cv2.COLOR_BGR2RGB)

        # Build the image columns dynamically — landmark view only appears
        # when the sidebar toggle is on, so the layout doesn't waste space
        # on a feature nobody asked for.
        image_labels = ["Face crop", "Grad-CAM heatmap"]
        image_arrays = [crop_rgb, overlay_rgb]
        if landmark_overlay is not None:
            mesh_bgr = landmark_overlay.draw(crop)
            image_arrays.append(cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2RGB))
            image_labels.append("Facial landmarks")

        img_cols = st.columns(len(image_arrays))
        for col, arr, label in zip(img_cols, image_arrays, image_labels):
            with col:
                st.image(arr, caption=label, use_container_width=True)

        info_col1, info_col2 = st.columns([1, 1])
        with info_col1:
            render_emotion_badge(emotion, confidence)
            st.markdown("**Full breakdown:**")
            for label, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                st.progress(score, text=f"{EMOTION_EMOJI.get(label, '')} {label}: {score * 100:.1f}%")

        with info_col2:
            if age_gender_estimator is not None:
                ag_result = age_gender_estimator.predict(crop)
                st.markdown("**Age & gender estimate**")
                st.markdown(
                    f"👤 **{ag_result['gender']}** ({ag_result['gender_confidence'] * 100:.0f}%) "
                    f"&nbsp;·&nbsp; 🎂 **~{ag_result['age_estimate']} yrs** "
                    f"({ag_result['age_range']}, {ag_result['age_confidence'] * 100:.0f}%)",
                    unsafe_allow_html=True,
                )
                st.caption("Age is an approximate midpoint of the model's predicted range, "
                           "not a precise estimate.")

            if old_classifier is not None:
                old_emotion, old_confidence, _ = classify_face(old_classifier, crop)
                st.markdown("**Model comparison**")
                st.markdown(
                    f"Original model: {EMOTION_EMOJI.get(old_emotion, '')} "
                    f"**{old_emotion.capitalize()}** ({old_confidence * 100:.1f}%)"
                )
                st.markdown(
                    f"Retrained model: {EMOTION_EMOJI.get(emotion, '')} "
                    f"**{emotion.capitalize()}** ({confidence * 100:.1f}%)"
                )
                delta = confidence - old_confidence
                if emotion == old_emotion:
                    st.caption(f"Same prediction — confidence changed by {delta:+.1%}")
                else:
                    st.caption("Different prediction between models — see confusion matrix "
                               "analysis for which one is more reliable on this class.")

        st.markdown("</div>", unsafe_allow_html=True)


if mode == "Image upload":
    st.caption("Upload one or more photos. Detects all faces, predicts emotion for each, "
               "shows confidence breakdown and a Grad-CAM heatmap.")

    uploaded_files = st.file_uploader(
        "Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) == 1:
            process_uploaded_image(Image.open(uploaded_files[0]).convert("RGB"))
        else:
            st.info(f"Processing {len(uploaded_files)} images — expand each to view results.")
            for file in uploaded_files:
                with st.expander(f"📷 {file.name}", expanded=False):
                    process_uploaded_image(Image.open(file).convert("RGB"))
    else:
        st.info("Upload an image above to get started.")


# ─────────────────────────────────────────────────────────────
# MODE 2 — Video file analysis
# ─────────────────────────────────────────────────────────────
elif mode == "Video file":
    st.caption("Upload a video. Samples frames across the clip, tracks emotion per "
               "person over time, and gives you a full session report.")

    sample_rate = st.slider("Sample every N seconds", 0.5, 5.0, 1.0, 0.5,
                             help="Lower = more accurate timeline, but slower to process.")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        # Peek at the video's duration up front so we can warn if the chosen
        # sample rate is longer than the clip itself — that combination
        # silently produces 0 or 1 predictions, which otherwise looks like
        # a bug rather than a sampling-rate mismatch.
        _probe_cap = cv2.VideoCapture(video_path)
        _probe_fps = _probe_cap.get(cv2.CAP_PROP_FPS) or 25
        _probe_frames = int(_probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _probe_cap.release()
        video_duration = _probe_frames / _probe_fps if _probe_fps else 0

        st.caption(f"Clip length: ~{video_duration:.1f} seconds ({_probe_frames} frames)")

        if sample_rate >= video_duration:
            st.warning(
                f"Sample rate ({sample_rate:.1f}s) is longer than this clip "
                f"(~{video_duration:.1f}s) — you'll get at most one prediction, "
                f"which won't be a reliable summary of the whole video. "
                f"Lower the sample rate below {video_duration:.1f}s for a real timeline."
            )
        elif video_duration / sample_rate < 5:
            st.info(
                f"This sample rate will only capture about "
                f"{int(video_duration / sample_rate)} frame(s) from this short clip. "
                f"Results (especially 'dominant emotion') will be noisy with so few "
                f"samples — consider lowering the sample rate for a more stable result."
            )

        analyze_clicked = st.button("Analyze video")
    else:
        analyze_clicked = False

    if uploaded_video is not None and analyze_clicked:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps * sample_rate))

        session = EmotionSession()
        tracker = CentroidTracker(max_disappeared=3, max_distance=150)
        smoother = EmotionSmoother(classifier.class_labels, window_size=5)
        progress = st.progress(0.0, text="Analyzing video...")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                faces = detector.detect(frame)
                boxes = [f["box"] for f in faces]
                tracked = tracker.update(boxes)

                # Match each tracked (id, box) back to the face it came from,
                # so we crop the right region for each persistent identity.
                for track_id, box in tracked.items():
                    matching_face = min(faces, key=lambda f: np.linalg.norm(
                        np.array(f["box"][:2]) - np.array(box[:2])))
                    crop = detector.crop(frame, matching_face, target_size=(224, 224))
                    if crop is None:
                        continue
                    _, _, raw_scores = classify_face(classifier, crop)
                    emotion, confidence, all_scores = smoother.update(track_id, raw_scores)

                    gender, age_range = None, None
                    if age_gender_estimator is not None:
                        ag_result = age_gender_estimator.predict(crop)
                        gender, age_range = ag_result["gender"], ag_result["age_range"]

                    session.add_record(timestamp, track_id, emotion, confidence, all_scores,
                                       gender=gender, age_range=age_range)

            frame_idx += 1
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        progress.empty()
        st.success(f"Analysis complete — {session.summary()['total_predictions']} predictions logged.")
        render_dashboard(session, key_prefix="video")


# ─────────────────────────────────────────────────────────────
# MODE 3 — Live webcam (multi-face, real-time)
# ─────────────────────────────────────────────────────────────
elif mode == "Live webcam":
    st.caption("Real-time multi-face emotion tracking from your webcam. "
               "Grad-CAM is disabled here for performance — it's designed for "
               "single-image deep-dives, not every frame of a live feed.")

    if "webcam_session" not in st.session_state:
        st.session_state.webcam_session = EmotionSession()
    if "webcam_tracker" not in st.session_state:
        st.session_state.webcam_tracker = CentroidTracker(max_disappeared=15, max_distance=150)
    if "webcam_smoother" not in st.session_state:
        st.session_state.webcam_smoother = EmotionSmoother(classifier.class_labels, window_size=8)
    if "webcam_age_gender_cache" not in st.session_state:
        st.session_state.webcam_age_gender_cache = {}  # track_id -> (gender, age_range), refreshed on each log tick

    run = st.checkbox("Start camera", key="webcam_run_checkbox")
    frame_placeholder = st.empty()
    dashboard_placeholder = st.container()

    if run:
        camera = cv2.VideoCapture(0)
        last_log_time = 0.0
        session = st.session_state.webcam_session
        tracker = st.session_state.webcam_tracker
        smoother = st.session_state.webcam_smoother
        ag_cache = st.session_state.webcam_age_gender_cache

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Could not read from webcam.")
                break

            if landmark_overlay is not None:
                frame = landmark_overlay.draw(frame)

            faces = detector.detect(frame)
            boxes = [f["box"] for f in faces]
            tracked = tracker.update(boxes)
            now = time.time() - session.start_time
            should_log = now - last_log_time >= webcam_log_interval

            for track_id, box in tracked.items():
                # Match the tracked box back to its detection (tracker only
                # returns boxes, not the full detection dict) so we can crop.
                matching_face = min(faces, key=lambda f: np.linalg.norm(
                    np.array(f["box"][:2]) - np.array(box[:2])))
                crop = detector.crop(frame, matching_face, target_size=(224, 224))
                if crop is None:
                    continue

                _, _, raw_scores = classify_face(classifier, crop)
                # Smooth over recent frames for this specific person — this
                # is what stops "happy" flickering to "neutral" and back
                # between consecutive frames of the same expression.
                emotion, confidence, all_scores = smoother.update(track_id, raw_scores)

                label_prefix = f"Person {track_id}"
                if age_gender_estimator is not None:
                    # Only re-run the age/gender model on the same throttled
                    # cadence as logging — every frame would needlessly cost
                    # two extra DNN forward passes per person per frame.
                    if should_log or track_id not in ag_cache:
                        ag_result = age_gender_estimator.predict(crop)
                        ag_cache[track_id] = (ag_result["gender"], ag_result["age_range"], ag_result["age_estimate"])
                    gender, age_range, age_estimate = ag_cache[track_id]
                    label_prefix = f"Person {track_id} ({gender}, ~{age_estimate})"

                frame = draw_label(frame, box, emotion, confidence, label_prefix=label_prefix)

                if should_log:
                    gender, age_range, _ = ag_cache.get(track_id, (None, None, None))
                    session.add_record(now, track_id, emotion, confidence, all_scores,
                                       gender=gender, age_range=age_range)

            if should_log:
                last_log_time = now

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Re-check the checkbox's current value each loop iteration —
            # this is what lets clicking "Start camera" off actually stop
            # the loop (Streamlit interrupts this run and starts a fresh
            # one with run=False, which then just skips the loop below).
            run = st.session_state.get("webcam_run_checkbox", run)

        camera.release()

    with dashboard_placeholder:
        session = st.session_state.webcam_session
        if session.summary()["total_predictions"] > 0:
            render_dashboard(session, key_prefix="webcam")
            if st.button("Clear session"):
                session.clear()
                st.session_state.webcam_tracker.reset()
                st.session_state.webcam_smoother.reset()
                st.session_state.webcam_age_gender_cache.clear()
                st.rerun()
        else:
            st.info("Start the camera to begin tracking.")


# ─────────────────────────────────────────────────────────────
# MODE 4 — History (past sessions, cross-session trends)
# ─────────────────────────────────────────────────────────────
elif mode == "History":
    st.caption("Every session you've saved (via the 'Save to history' button in Video/Webcam "
               "mode) lives here — so you can compare satisfaction trends across days, not just "
               "look at one session in isolation.")

    sessions_df = history.list_sessions()

    if sessions_df.empty:
        st.info("No saved sessions yet. Run a Video or Webcam session, then click "
                "'💾 Save to history' on its dashboard to start building history.")
    else:
        st.markdown('<div class="deepfer-card">', unsafe_allow_html=True)
        st.markdown("**Saved sessions**")
        st.dataframe(sessions_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        trend_df = history.get_trend_data()
        if len(trend_df) > 1:
            st.markdown("**Satisfaction score over time (across sessions)**")
            trend_chart_df = trend_df.set_index("created_at")[["satisfaction_score"]]
            st.line_chart(trend_chart_df)

        if len(sessions_df) >= 2:
            st.markdown("---")
            st.markdown("### 🆚 Compare two sessions")

            def _session_option_label(i):
                row = sessions_df.loc[sessions_df["id"] == i].iloc[0]
                return f"#{i} — {row['label'] or row['source']} ({row['created_at'][:16]})"

            compare_col1, compare_col2 = st.columns(2)
            with compare_col1:
                session_a_id = st.selectbox(
                    "Session A", sessions_df["id"].tolist(),
                    format_func=_session_option_label, key="compare_session_a",
                )
            with compare_col2:
                remaining_ids = [i for i in sessions_df["id"].tolist() if i != session_a_id]
                session_b_id = st.selectbox(
                    "Session B", remaining_ids,
                    format_func=_session_option_label, key="compare_session_b",
                )

            session_a = history.get_session(session_a_id)
            session_b = history.get_session(session_b_id)

            metric_col1, metric_col2 = st.columns(2)
            for col, sess in [(metric_col1, session_a), (metric_col2, session_b)]:
                with col:
                    st.markdown(f'<div class="deepfer-card">', unsafe_allow_html=True)
                    st.markdown(f"**Session #{sess['id']}** — {sess['label'] or sess['source']}")
                    st.metric("Total predictions", sess["total_predictions"])
                    dom = sess["dominant_emotion"]
                    st.metric("Dominant emotion", f"{EMOTION_EMOJI.get(dom, '')} {dom.capitalize() if dom else '—'}")
                    st.metric("Satisfaction score", f"{sess['satisfaction_score']:+.1f}")
                    dist_df = pd.DataFrame(
                        list(sess["emotion_distribution"].items()), columns=["Emotion", "Percent"]
                    ).set_index("Emotion")
                    st.bar_chart(dist_df)
                    st.markdown("</div>", unsafe_allow_html=True)

            score_delta = session_b["satisfaction_score"] - session_a["satisfaction_score"]
            if abs(score_delta) < 0.1:
                st.info("Both sessions have essentially the same satisfaction score.")
            else:
                better = "Session B" if score_delta > 0 else "Session A"
                st.success(f"{better} had the higher satisfaction score, by {abs(score_delta):.1f} points.")

        st.markdown("---")
        st.markdown("**Inspect or remove a session**")
        selected_id = st.selectbox(
            "Session ID", sessions_df["id"].tolist(),
            format_func=lambda i: f"#{i} — {sessions_df.loc[sessions_df['id'] == i, 'label'].values[0] or sessions_df.loc[sessions_df['id'] == i, 'source'].values[0]}",
        )

        col_view, col_delete = st.columns([3, 1])
        with col_view:
            if st.button("View details"):
                records = history.load_session_records(selected_id)
                if not records.empty:
                    st.dataframe(records, use_container_width=True, hide_index=True)
                else:
                    st.warning("No detailed records found for this session.")
        with col_delete:
            if st.button("🗑️ Delete session", key="delete_session_btn"):
                history.delete_session(selected_id)
                st.success(f"Deleted session #{selected_id}.")
                st.rerun()
