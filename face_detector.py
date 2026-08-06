"""
face_detector.py — Face detection module for DeepFER.

Finds faces in an image/frame and returns cropped, padded face regions
ready to feed into the emotion classifier.

Uses MediaPipe Face Detection as the primary method (fast, accurate, works
well across angles/lighting). Falls back to OpenCV's Haar Cascade if
MediaPipe isn't installed, so this still works with just opencv-python.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


class FaceDetector:
    """Detects faces in an image and returns cropped face regions.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(frame)   # list of FaceBox
        for face in faces:
            crop = detector.crop(frame, face, target_size=(224, 224))
    """

    def __init__(self, min_confidence=0.5, padding_ratio=0.25):
        """
        Args:
            min_confidence: minimum detection confidence (0-1) to keep a face.
            padding_ratio: extra margin added around each detected face box,
                as a fraction of the box size. Emotion recognition benefits
                from including a bit of forehead/chin/ears, not just a tight
                crop of the eyes-nose-mouth region.
        """
        self.min_confidence = min_confidence
        self.padding_ratio = padding_ratio
        self._backend = "mediapipe" if _MEDIAPIPE_AVAILABLE else "opencv"

        if self._backend == "mediapipe":
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 1 = full-range model, better for varied distances
                min_detection_confidence=min_confidence,
            )
        else:
            # Bundled with opencv-python — no extra download needed.
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)

    @property
    def backend(self):
        """Which detector is actually active: 'mediapipe' or 'opencv'."""
        return self._backend

    def detect(self, frame_bgr):
        """Detect faces in a BGR image (as read by cv2.imread / cv2.VideoCapture).

        Returns:
            List of dicts: [{'box': (x, y, w, h), 'confidence': float}, ...]
            Boxes are in pixel coordinates, sorted left-to-right.
        """
        h, w = frame_bgr.shape[:2]
        results = []

        if self._backend == "mediapipe":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = self._mp_face.process(rgb)
            if detections.detections:
                for det in detections.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    box_w = int(bbox.width * w)
                    box_h = int(bbox.height * h)
                    confidence = det.score[0] if det.score else 0.0
                    results.append({"box": (x, y, box_w, box_h), "confidence": confidence})
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, box_w, box_h) in faces:
                # Haar cascade doesn't give a confidence score; treat all as 1.0
                results.append({"box": (x, y, box_w, box_h), "confidence": 1.0})

        results = [r for r in results if r["confidence"] >= self.min_confidence]
        results.sort(key=lambda r: r["box"][0])  # left to right
        return results

    def crop(self, frame_bgr, face, target_size=(224, 224)):
        """Crop a detected face out of the frame, with padding, resized for the model.

        Args:
            frame_bgr: the full original image.
            face: one entry from detect() — a dict with a 'box' key.
            target_size: (width, height) to resize the crop to, matching
                the emotion model's expected input (224x224 for DeepFER).

        Returns:
            Cropped, resized BGR image, or None if the box is degenerate.
        """
        h, w = frame_bgr.shape[:2]
        x, y, box_w, box_h = face["box"]

        pad_w = int(box_w * self.padding_ratio)
        pad_h = int(box_h * self.padding_ratio)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + box_w + pad_w)
        y2 = min(h, y + box_h + pad_h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        return crop

    def close(self):
        """Release resources (call when done, especially for webcam loops)."""
        if self._backend == "mediapipe":
            self._mp_face.close()
