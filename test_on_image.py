"""
test_on_image.py — Quick end-to-end test: detect faces in an image, predict
emotion for each, and save an annotated output image.

Usage:
    python test_on_image.py path/to/photo.jpg

Requires (install once):
    pip install opencv-python mediapipe tensorflow
"""

import sys
import numpy as np
import cv2
from PIL import Image
from face_detector import FaceDetector
from emotion_inference import EmotionClassifier

MODEL_PATH = "deepfer_mobilenetv2_v2.h5"
LABELS_PATH = "class_labels.json"


def load_image_any_format(path):
    """Load an image via Pillow (handles JPEG, PNG, HEIC, WEBP, and files with
    a mismatched extension) and convert it to an OpenCV-style BGR array.
    More robust than cv2.imread, which only handles a narrower set of formats
    and fails silently (returns None) on anything it doesn't recognize."""
    pil_img = Image.open(path).convert("RGB")
    rgb_array = np.array(pil_img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_on_image.py path/to/photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        frame = load_image_any_format(image_path)
    except Exception as e:
        print(f"Could not read image: {image_path}")
        print(f"  Error: {e}")
        sys.exit(1)

    print("Loading face detector...")
    detector = FaceDetector()
    print(f"  Using backend: {detector.backend}")

    print("Loading emotion model...")
    classifier = EmotionClassifier(MODEL_PATH, LABELS_PATH)

    print("Detecting faces...")
    faces = detector.detect(frame)
    print(f"  Found {len(faces)} face(s).")

    if len(faces) == 0:
        print("No faces detected. Try a clearer, front-facing photo.")
        return

    for i, face in enumerate(faces):
        crop = detector.crop(frame, face)
        if crop is None:
            continue

        result = classifier.predict(crop)
        x, y, w, h = face["box"]

        print(f"\nFace {i + 1}:")
        print(f"  Predicted emotion: {result['emotion']} ({result['confidence']:.1%} confidence)")
        print("  Full breakdown:")
        for label, score in sorted(result["all_scores"].items(), key=lambda kv: -kv[1]):
            print(f"    {label:10s} {score:.1%}")

        # Draw box + label on the output image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = f"{result['emotion']} ({result['confidence']:.0%})"
        cv2.putText(
            frame, label_text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    output_path = "output_annotated.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\nAnnotated image saved to: {output_path}")

    detector.close()


if __name__ == "__main__":
    main()
