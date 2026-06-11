"""
emotion_engine.py
Handles facial emotion detection, preprocessing, smoothing, and temporal stabilization.
"""

import cv2
import numpy as np
from collections import deque, Counter
import time

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARN] DeepFace not installed. Using mock emotion detection.")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
STABILIZATION_WINDOW   = 5.0   # seconds
MOVING_AVG_WINDOW      = 8     # frames for smoothing confidence
MIN_CONFIDENCE         = 0.30  # ignore detections below this threshold

# Raw DeepFace emotion → grouped emotion
EMOTION_MAP = {
    "happy":    "Happy",
    "sad":      "Sad",
    "angry":    "Angry",
    "fear":     "Sad",
    "disgust":  "Angry",
    "surprise": "Happy",
    "neutral":  "Neutral",
}

# ─────────────────────────────────────────────
# Gamma correction for low-light improvement
# ─────────────────────────────────────────────
def gamma_correction(frame: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(frame, table)


# ─────────────────────────────────────────────
# Emotion Engine
# ─────────────────────────────────────────────
class EmotionEngine:
    def __init__(self):
        self.raw_emotion_history: deque = deque(maxlen=30)      # for majority voting
        self.confidence_history:  deque = deque(maxlen=MOVING_AVG_WINDOW)

        self.stable_emotion:  str   = "Neutral"
        self.last_commit_time: float = time.time()
        self.current_emotion:  str   = "Neutral"
        self.current_confidence: float = 0.0

        # Mock state for when DeepFace is unavailable
        self._mock_emotions = ["Happy", "Sad", "Angry", "Neutral"]
        self._mock_idx = 0
        self._mock_counter = 0

    # ── preprocessing ──────────────────────────────────────────
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return gamma_correction(frame, gamma=1.4)

    # ── single-frame detection ──────────────────────────────────
    def detect_emotion(self, frame: np.ndarray) -> tuple[str | None, float]:
        """
        Returns (raw_emotion_label, confidence) or (None, 0.0) on failure.
        """
        if not DEEPFACE_AVAILABLE:
            return self._mock_detect()

        try:
            processed = self.preprocess(frame)
            results = DeepFace.analyze(
                processed,
                actions=["emotion"],
                enforce_detection=True,
                detector_backend="opencv",
                silent=True,
            )
            result = results[0] if isinstance(results, list) else results
            emotions: dict = result["emotion"]           # {label: score, ...}
            dominant: str  = result["dominant_emotion"]

            # Convert raw scores to probabilities (softmax-like normalisation)
            total = sum(emotions.values())
            if total == 0:
                return None, 0.0
            confidence = round(emotions[dominant] / total, 2)
            return dominant.lower(), confidence

        except Exception:
            return None, 0.0

    # ── moving-average confidence ───────────────────────────────
    def smooth_confidence(self, conf: float) -> float:
        self.confidence_history.append(conf)
        return round(float(np.mean(self.confidence_history)), 2)

    # ── temporal stabilization via majority vote ────────────────
    def stabilize(self, raw_emotion: str | None, confidence: float) -> tuple[str, float]:
        """
        Accumulates raw detections; returns the stable emotion only after
        the stabilization window has elapsed and majority vote agrees.
        """
        smoothed_conf = self.smooth_confidence(confidence)

        if raw_emotion is None or smoothed_conf < MIN_CONFIDENCE:
            self.current_confidence = smoothed_conf
            return self.stable_emotion, smoothed_conf

        # Map to grouped category
        grouped = EMOTION_MAP.get(raw_emotion, "Neutral")
        self.raw_emotion_history.append(grouped)
        self.current_confidence = smoothed_conf

        now = time.time()
        elapsed = now - self.last_commit_time

        if elapsed >= STABILIZATION_WINDOW:
            # Majority vote over the accumulated window
            if self.raw_emotion_history:
                most_common, _ = Counter(self.raw_emotion_history).most_common(1)[0]
                self.stable_emotion   = most_common
                self.last_commit_time = now
                self.raw_emotion_history.clear()

        self.current_emotion = grouped
        return self.stable_emotion, smoothed_conf

    # ── process a full frame ────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> tuple[str, float]:
        raw, conf = self.detect_emotion(frame)
        return self.stabilize(raw, conf)

    # ── annotate frame for streaming ────────────────────────────
    def annotate_frame(
        self,
        frame: np.ndarray,
        stable_emotion: str,
        confidence: float,
    ) -> np.ndarray:
        label = f"{stable_emotion}  ({confidence:.2f})"
        cv2.putText(
            frame, label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1, (0, 255, 128), 2, cv2.LINE_AA,
        )
        return frame

    # ── mock detection (no DeepFace) ────────────────────────────
    def _mock_detect(self) -> tuple[str, float]:
        self._mock_counter += 1
        if self._mock_counter % 60 == 0:          # change every ~60 frames
            self._mock_idx = (self._mock_idx + 1) % len(self._mock_emotions)
        emotion = self._mock_emotions[self._mock_idx].lower()
        confidence = round(0.70 + 0.05 * np.random.randn(), 2)
        confidence = max(0.0, min(1.0, confidence))
        return emotion, confidence
