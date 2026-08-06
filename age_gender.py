"""
age_gender.py — Age and gender estimation for DeepFER.

Uses pretrained Caffe models (Levi & Hassner architecture, via the widely
used LearnOpenCV distribution) rather than training a new model from
scratch — training a proper age/gender model needs its own dataset
(e.g. UTKFace) and a multi-hour Kaggle training run, which is a separate
project on its own. This gets solid, real predictions with zero training.

Download the four required model files into a `models/` folder next to
this script:

    age_deploy.prototxt
        https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt
    age_net.caffemodel
        https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/age_net.caffemodel
    gender_deploy.prototxt
        https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt
    gender_net.caffemodel
        https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel

(If any link ever moves, search "age_net.caffemodel gender_net.caffemodel
download" — these are standard, widely-mirrored files.)
"""

import os
import cv2
import numpy as np

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LABELS = ["Male", "Female"]

# Midpoint of each bucket, for an approximate single-number age display.
# IMPORTANT: this is NOT added precision — the underlying model only ever
# picks one of the 8 buckets above; this just relabels that same bucket
# as one number instead of a range. Always present it as "~28" or
# "approx. 28", never as a bare exact number, so it doesn't imply
# precision the model doesn't have.
AGE_BUCKET_MIDPOINTS = [1, 5, 10, 17, 28, 40, 50, 75]


class AgeGenderEstimator:
    """Wraps the pretrained Caffe age/gender models for face-crop inference."""

    def __init__(self, model_dir="models"):
        """
        Args:
            model_dir: folder containing the four Caffe model files listed
                in this module's docstring.
        """
        age_proto = os.path.join(model_dir, "age_deploy.prototxt")
        age_model = os.path.join(model_dir, "age_net.caffemodel")
        gender_proto = os.path.join(model_dir, "gender_deploy.prototxt")
        gender_model = os.path.join(model_dir, "gender_net.caffemodel")

        for path in (age_proto, age_model, gender_proto, gender_model):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing age/gender model file: {path}\n"
                    "See the download links in age_gender.py's module docstring."
                )

        self.age_net = cv2.dnn.readNet(age_model, age_proto)
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)

    def predict(self, face_crop_bgr):
        """Predict age bucket and gender for a single face crop.

        Args:
            face_crop_bgr: a cropped face image (any size — internally
                resized to what the models expect).

        Returns:
            dict with:
                'gender': 'Male' or 'Female'
                'gender_confidence': float, 0-1
                'age_range': e.g. '(25-32)' — the model's actual output
                'age_estimate': e.g. 28 — bucket midpoint, an approximation
                    for display purposes only, not genuine added precision
                'age_confidence': float, 0-1
        """
        blob = cv2.dnn.blobFromImage(
            face_crop_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False,
        )

        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()[0]
        gender_idx = int(np.argmax(gender_preds))

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()[0]
        age_idx = int(np.argmax(age_preds))

        return {
            "gender": GENDER_LABELS[gender_idx],
            "gender_confidence": float(gender_preds[gender_idx]),
            "age_range": AGE_BUCKETS[age_idx],
            "age_estimate": AGE_BUCKET_MIDPOINTS[age_idx],
            "age_confidence": float(age_preds[age_idx]),
        }
