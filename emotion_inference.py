"""
emotion_inference.py — Loads the trained DeepFER model and predicts emotions
on faces detected in an image or video frame.

Usage:
    from face_detector import FaceDetector
    from emotion_inference import EmotionClassifier

    detector = FaceDetector()
    classifier = EmotionClassifier("deepfer_mobilenetv2.h5", "class_labels.json")

    frame = cv2.imread("photo.jpg")
    faces = detector.detect(frame)
    for face in faces:
        crop = detector.crop(frame, face)
        result = classifier.predict(crop)
        print(result)  # {'emotion': 'happy', 'confidence': 0.82, 'all_scores': {...}}
"""

import json
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class EmotionClassifier:
    """Wraps the trained DeepFER model for single-face emotion prediction."""

    def __init__(self, model_path, class_labels_path):
        """
        Args:
            model_path: path to deepfer_mobilenetv2.h5
            class_labels_path: path to class_labels.json (the 7 emotion names,
                in the exact order the model outputs them)
        """
        self.model = keras.models.load_model(model_path, compile=False)

        with open(class_labels_path, "r") as f:
            self.class_labels = json.load(f)

        # Confirm the model's output layer size matches the label count —
        # catches a mismatched model/labels pairing early with a clear error
        # instead of silent misclassification.
        output_size = self.model.output_shape[-1]
        if output_size != len(self.class_labels):
            raise ValueError(
                f"Model outputs {output_size} classes but class_labels.json "
                f"has {len(self.class_labels)} entries — check you're using "
                f"the matching pair of files."
            )

    def predict(self, face_crop_bgr):
        """Predict the emotion for a single cropped face image.

        Args:
            face_crop_bgr: a 224x224x3 BGR image (as returned by
                FaceDetector.crop()). Any size will be resized automatically.

        Returns:
            dict with:
                'emotion': the top predicted label (str)
                'confidence': the top prediction's probability (float, 0-1)
                'all_scores': dict of every class -> probability, for
                    displaying a full breakdown (not just the top guess)
        """
        import cv2

        img = cv2.resize(face_crop_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb.astype(np.float32), axis=0)
        img_array = preprocess_input(img_array)  # must match training preprocessing

        predictions = self.model.predict(img_array, verbose=0)[0]

        all_scores = {
            label: float(score)
            for label, score in zip(self.class_labels, predictions)
        }
        top_idx = int(np.argmax(predictions))

        return {
            "emotion": self.class_labels[top_idx],
            "confidence": float(predictions[top_idx]),
            "all_scores": all_scores,
        }
