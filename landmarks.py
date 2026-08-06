"""
landmarks.py — Facial landmark overlay using MediaPipe FaceMesh.

Draws the 468-point face mesh (tessellation + contours: eyes, eyebrows,
lips, face oval) directly onto a frame — purely visual/explanatory, this
doesn't feed into the emotion prediction at all. It's what makes the app
look like it's doing detailed facial analysis, not just a black-box
classifier.
"""

import cv2
import mediapipe as mp

_mp_face_mesh = mp.solutions.face_mesh
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


class LandmarkOverlay:
    """Detects and draws facial landmarks on frames/crops."""

    def __init__(self, max_faces=5, min_detection_confidence=0.5):
        self._mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # adds iris landmarks too
            min_detection_confidence=min_detection_confidence,
        )

    def draw(self, frame_bgr):
        """Detect landmarks and draw them on a copy of frame_bgr.

        Returns:
            Annotated BGR frame (same size as input). If no face is found,
            returns an unmodified copy — never raises on a landmark-less frame.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        annotated = frame_bgr.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                _mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=_mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                _mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=_mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

        return annotated

    def close(self):
        self._mesh.close()
