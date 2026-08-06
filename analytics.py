"""
analytics.py — Session-level emotion tracking for DeepFER.

Collects predictions over time (from a video or live webcam session) and
turns them into summary statistics, a satisfaction score (for the customer
service framing), and an exportable CSV — turning the raw classifier into
an actual analytics tool instead of a one-shot predictor.
"""

import time
import io
import pandas as pd


# Maps raw emotion classes to a 3-way "customer satisfaction" framing.
# This is the layer that turns "happy/sad/angry" into a business-relevant
# signal — e.g. for a customer service kiosk or classroom engagement view.
SATISFACTION_MAP = {
    "happy": "Satisfied",
    "surprise": "Satisfied",
    "neutral": "Neutral",
    "sad": "Dissatisfied",
    "angry": "Dissatisfied",
    "fear": "Dissatisfied",
    "disgust": "Dissatisfied",
}


class EmotionSession:
    """Accumulates emotion predictions over a session (video or live feed)."""

    def __init__(self):
        self.records = []  # list of dicts: timestamp, face_id, emotion, confidence, all_scores
        self.start_time = time.time()

    def add_record(self, timestamp_seconds, face_id, emotion, confidence, all_scores,
                    gender=None, age_range=None):
        """Log one prediction.

        Args:
            timestamp_seconds: seconds since session start (for webcam) or
                seconds into the clip (for video file analysis).
            face_id: which face in the frame this belongs to (0, 1, 2...),
                since multiple people can be tracked in the same frame.
            emotion: the predicted top emotion label.
            confidence: the top emotion's probability (0-1).
            all_scores: dict of every class -> probability.
            gender: optional — 'Male'/'Female' from AgeGenderEstimator, if used.
            age_range: optional — e.g. '(25-32)' from AgeGenderEstimator, if used.
        """
        self.records.append({
            "timestamp": round(timestamp_seconds, 2),
            "face_id": face_id,
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "satisfaction": SATISFACTION_MAP.get(emotion, "Neutral"),
            "gender": gender,
            "age_range": age_range,
            **{f"score_{label}": round(score, 4) for label, score in all_scores.items()},
        })

    def to_dataframe(self):
        """Return all records as a pandas DataFrame, empty-safe."""
        if not self.records:
            return pd.DataFrame(columns=["timestamp", "face_id", "emotion", "confidence", "satisfaction"])
        return pd.DataFrame(self.records)

    def summary(self):
        """Compute headline stats for the dashboard.

        Returns:
            dict with:
                total_predictions: int
                emotion_distribution: {emotion: percentage}
                satisfaction_distribution: {Satisfied/Neutral/Dissatisfied: percentage}
                satisfaction_score: float, -100 to +100
                    (% Satisfied - % Dissatisfied; Neutral doesn't count
                    either way — a simple, explainable net score, similar
                    to Net Promoter Score logic)
                dominant_emotion: the most common emotion overall
        """
        df = self.to_dataframe()
        if df.empty:
            return {
                "total_predictions": 0,
                "emotion_distribution": {},
                "satisfaction_distribution": {},
                "satisfaction_score": 0.0,
                "dominant_emotion": None,
            }

        emotion_counts = df["emotion"].value_counts(normalize=True) * 100
        satisfaction_counts = df["satisfaction"].value_counts(normalize=True) * 100

        pct_satisfied = satisfaction_counts.get("Satisfied", 0.0)
        pct_dissatisfied = satisfaction_counts.get("Dissatisfied", 0.0)
        satisfaction_score = pct_satisfied - pct_dissatisfied

        return {
            "total_predictions": len(df),
            "emotion_distribution": emotion_counts.round(1).to_dict(),
            "satisfaction_distribution": satisfaction_counts.round(1).to_dict(),
            "satisfaction_score": round(satisfaction_score, 1),
            "dominant_emotion": df["emotion"].mode().iloc[0],
        }

    def notable_moments(self, threshold_confidence=0.75):
        """Find high-confidence spikes of non-neutral emotion — the kind of
        thing worth calling out in a report, e.g. 'spike of anger at 1:20'.

        Returns a list of dicts: {timestamp, emotion, confidence}, sorted
        by timestamp, limited to strong/interesting moments.
        """
        df = self.to_dataframe()
        if df.empty:
            return []

        interesting = df[
            (df["confidence"] >= threshold_confidence) & (df["emotion"] != "neutral")
        ].sort_values("timestamp")

        return interesting[["timestamp", "face_id", "emotion", "confidence"]].to_dict("records")

    def to_csv_bytes(self):
        """Export the full session log as CSV bytes, ready for st.download_button."""
        df = self.to_dataframe()
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")

    def clear(self):
        self.records = []
        self.start_time = time.time()
