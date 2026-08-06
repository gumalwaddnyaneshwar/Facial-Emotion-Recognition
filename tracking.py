"""
tracking.py — Persistent face identity tracking + temporal emotion smoothing.

Two problems this solves:

1. Without tracking, "face_id" is just index order per frame — if two
   people move around or one briefly leaves frame, "Face 0" can silently
   become a different person. CentroidTracker assigns stable IDs by
   matching face positions across frames, so "Person 1" stays Person 1.

2. Raw per-frame predictions flicker — someone can be predicted "happy"
   for one frame and "neutral" the next even mid-expression, because the
   model has no memory of prior frames. EmotionSmoother keeps a short
   rolling window of each tracked person's recent score vectors and
   averages them, so predictions change only when the underlying
   expression genuinely shifts.
"""

from collections import OrderedDict, deque
import numpy as np


class CentroidTracker:
    """Assigns and maintains persistent IDs for faces across frames.

    Matches new detections to existing tracked faces by centroid distance
    (nearest-neighbor greedy matching) — simple, fast, and good enough for
    a handful of faces in a webcam/video frame (not designed for crowded
    scenes with dozens of people).

    IDs are only revealed to the caller once a track has been seen for
    `min_hits` consecutive frames — this stops brief false-positive
    detections (background clutter, motion blur, a passing reflection)
    from consuming a "Person N" number that then never reappears. Without
    this, a session could jump straight to "Person 3" despite only one
    real face ever being on screen, because 3 phantom one-frame blips
    used up IDs 0-2 first.
    """

    def __init__(self, max_disappeared=15, max_distance=150, min_hits=3):
        """
        Args:
            max_disappeared: how many consecutive frames a tracked face can
                go undetected before its ID is dropped (handles brief
                occlusion/looking away without losing identity immediately).
            max_distance: max pixel distance between frames for a detection
                to be considered the same person. Too low = IDs reassign
                when people move normally; too high = two nearby people can
                get swapped. Tune based on your frame resolution.
            min_hits: how many consecutive frames a candidate face must be
                detected before it's assigned a visible display ID. Higher
                = fewer phantom IDs from false-positive blips, but a very
                brief real appearance may go unlabeled. 3 is roughly a
                fraction of a second at normal webcam frame rates.
        """
        self.next_candidate_id = 0
        self.next_display_id = 0
        self.objects = OrderedDict()        # candidate_id -> box (x, y, w, h)
        self.disappeared = OrderedDict()    # candidate_id -> consecutive missed frames
        self.hits = OrderedDict()           # candidate_id -> consecutive frames seen
        self.display_ids = OrderedDict()    # candidate_id -> display_id (once confirmed)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits

    @staticmethod
    def _centroid(box):
        x, y, w, h = box
        return np.array([x + w / 2.0, y + h / 2.0])

    def _register(self, box):
        candidate_id = self.next_candidate_id
        self.objects[candidate_id] = box
        self.disappeared[candidate_id] = 0
        self.hits[candidate_id] = 1
        self.next_candidate_id += 1
        return candidate_id

    def _deregister(self, candidate_id):
        del self.objects[candidate_id]
        del self.disappeared[candidate_id]
        del self.hits[candidate_id]
        self.display_ids.pop(candidate_id, None)

    def _confirm_if_ready(self, candidate_id):
        """Assign a visible display ID once a candidate crosses min_hits —
        called only on frames where the candidate was actually matched,
        so a string of misses can't sneak it past the threshold.
        """
        if candidate_id not in self.display_ids and self.hits[candidate_id] >= self.min_hits:
            self.display_ids[candidate_id] = self.next_display_id
            self.next_display_id += 1

    def update(self, boxes):
        """Update tracks with this frame's detected face boxes.

        Args:
            boxes: list of (x, y, w, h) tuples, one per detected face.

        Returns:
            OrderedDict of {display_id: box} for confirmed, currently
            visible faces only. Candidates still building up hits, or
            faces that disappeared, aren't included — but confirmed
            identities are preserved through brief disappearances (up to
            max_disappeared frames) in case they reappear.
        """
        if len(boxes) == 0:
            # No detections this frame — age out any tracks that have been
            # missing too long.
            for candidate_id in list(self.disappeared.keys()):
                self.disappeared[candidate_id] += 1
                if self.disappeared[candidate_id] > self.max_disappeared:
                    self._deregister(candidate_id)
            return OrderedDict()

        input_centroids = np.array([self._centroid(b) for b in boxes])

        if len(self.objects) == 0:
            for box in boxes:
                candidate_id = self._register(box)
                self._confirm_if_ready(candidate_id)
        else:
            candidate_ids = list(self.objects.keys())
            object_centroids = np.array([self._centroid(b) for b in self.objects.values()])

            # Pairwise distance matrix: existing tracks x new detections
            D = np.linalg.norm(
                object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
                axis=2,
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                candidate_id = candidate_ids[row]
                self.objects[candidate_id] = boxes[col]
                self.disappeared[candidate_id] = 0
                self.hits[candidate_id] += 1
                self._confirm_if_ready(candidate_id)
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                candidate_id = candidate_ids[row]
                self.disappeared[candidate_id] += 1
                if self.disappeared[candidate_id] > self.max_disappeared:
                    self._deregister(candidate_id)

            for col in unused_cols:
                candidate_id = self._register(boxes[col])
                self._confirm_if_ready(candidate_id)

        return OrderedDict(
            (self.display_ids[cid], box)
            for cid, box in self.objects.items()
            if cid in self.display_ids
        )

    def reset(self):
        self.next_candidate_id = 0
        self.next_display_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.hits.clear()
        self.display_ids.clear()


class EmotionSmoother:
    """Smooths per-person emotion predictions over a rolling window of
    recent frames, keyed by the tracked person's ID.
    """

    def __init__(self, class_labels, window_size=8):
        """
        Args:
            class_labels: ordered list of emotion labels (must match the
                order used in all_scores dicts from EmotionClassifier).
            window_size: how many recent frames to average over. Larger =
                smoother but slower to react to real expression changes;
                8 frames at ~throttled 1/sec logging is a few seconds of
                context, which is a reasonable default.
        """
        self.class_labels = class_labels
        self.window_size = window_size
        self.history = {}  # track_id -> deque of score arrays

    def update(self, track_id, all_scores):
        """Feed in this frame's raw prediction for a tracked person.

        Args:
            track_id: the persistent ID from CentroidTracker.
            all_scores: dict of {label: probability} for this frame.

        Returns:
            (smoothed_emotion, smoothed_confidence, smoothed_all_scores)
            — same shape as a normal prediction, just averaged over the
            recent window for that specific person.
        """
        vector = np.array([all_scores[label] for label in self.class_labels])

        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)
        self.history[track_id].append(vector)

        avg_vector = np.mean(self.history[track_id], axis=0)
        top_idx = int(np.argmax(avg_vector))

        smoothed_scores = {label: float(score) for label, score in zip(self.class_labels, avg_vector)}
        return self.class_labels[top_idx], float(avg_vector[top_idx]), smoothed_scores

    def forget(self, track_id):
        """Clear history for a track that's no longer active (e.g. person left frame)."""
        self.history.pop(track_id, None)

    def reset(self):
        self.history.clear()
