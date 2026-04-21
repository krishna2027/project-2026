"""
Sign Language Translator – Flask Application
=============================================

Routes
------
GET  /              – Serve the main UI page.
GET  /video_feed    – MJPEG stream with hand-tracking overlay.
GET  /api/state     – Current app state (JSON, polled by the frontend).
POST /api/clear     – Clear the typed text buffer.
POST /api/backspace – Remove the last character.
POST /api/space     – Append a space character.

MediaPipe hand detection
------------------------
This app uses the MediaPipe Tasks API (mediapipe 0.10+).
It requires the hand landmarker model bundle:

    models/hand_landmarker.task

Download it once before running:

    python scripts/download_models.py

or manually from:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Set HAND_LANDMARKER_MODEL env var to a custom path if needed.
"""

import logging
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from model.sign_language_model import SignLanguageModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# MediaPipe Tasks API (optional – requires hand_landmarker.task bundle)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "hand_landmarker.task"
)
HAND_MODEL_PATH = os.environ.get("HAND_LANDMARKER_MODEL", _DEFAULT_MODEL_PATH)

# Hand connection pairs (MediaPipe landmark indices)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]


def _try_init_hand_landmarker():
    """
    Attempt to build a MediaPipe HandLandmarker.

    Returns the landmarker object on success, or None if the model bundle
    file does not exist or MediaPipe fails to initialise.
    """
    if not os.path.exists(HAND_MODEL_PATH):
        logger.warning(
            "Hand landmarker model not found at %s. "
            "Hand tracking will be disabled. "
            "Run  python scripts/download_models.py  to download it.",
            HAND_MODEL_PATH,
        )
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base

        BaseOptions = mp_base.BaseOptions
        HandLandmarker = mp_vision.HandLandmarker
        HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
        RunningMode = mp_vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker initialised from %s", HAND_MODEL_PATH)
        return landmarker
    except Exception as exc:
        logger.error("Failed to initialise HandLandmarker: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

sign_model = SignLanguageModel()
logger.info("Sign model: %r", sign_model)

# ---------------------------------------------------------------------------
# Global shared state (protected by a lock)
# ---------------------------------------------------------------------------


class AppState:
    """Holds the mutable state shared between the video thread and HTTP routes."""

    # How many consecutive frames the *same* sign must be detected before it
    # is accepted (≈1 second at 30 fps).
    STABILITY_THRESHOLD: int = 30

    # Cooldown frames after a sign is accepted so the same letter is not
    # registered twice in quick succession (≈1.5 s at 30 fps).
    COOLDOWN_THRESHOLD: int = 45

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_sign: str = "NOTHING"
        self.confidence: float = 0.0
        self.typed_text: str = ""
        self.stability_count: int = 0
        self.cooldown_frames: int = 0
        self.hand_detected: bool = False


state = AppState()

# ---------------------------------------------------------------------------
# Landmark drawing helper
# ---------------------------------------------------------------------------

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_CYAN   = (0, 212, 255)
_GREEN  = (16, 185, 129)
_ORANGE = (255, 165, 0)
_WHITE  = (255, 255, 255)
_DARK   = (20, 20, 40)


def _draw_landmarks(frame: np.ndarray, landmarks: list) -> None:
    """Draw hand landmark skeleton onto *frame* (in-place)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Connections
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 120), 2, cv2.LINE_AA)

    # Landmark dots
    for i, (x, y) in enumerate(pts):
        r = 5 if i == 0 else 4
        cv2.circle(frame, (x, y), r, _CYAN, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r + 1, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_overlay(
    frame: np.ndarray,
    current_sign: str,
    confidence: float,
    stability: float,
    is_trained: bool,
    in_cooldown: bool,
    hand_tracking_available: bool,
) -> None:
    """Draw the informational HUD on the camera frame (in-place)."""
    h, w = frame.shape[:2]
    bar_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), _DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Current sign label ─────────────────────────────────────────────
    label = current_sign if current_sign != "NOTHING" else "No sign"
    cv2.putText(frame, f"Sign: {label}", (12, h - bar_h + 26),
                _FONT, 0.8, _CYAN, 2, cv2.LINE_AA)

    # ── Confidence bar ─────────────────────────────────────────────────
    if confidence > 0:
        bar_w = int(confidence * 220)
        cv2.rectangle(frame, (12, h - bar_h + 38), (232, h - bar_h + 53),
                      (60, 60, 60), -1)
        color = _GREEN if confidence > 0.75 else _ORANGE
        cv2.rectangle(frame, (12, h - bar_h + 38),
                      (12 + bar_w, h - bar_h + 53), color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (240, h - bar_h + 51),
                    _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)

    # ── Stability bar ──────────────────────────────────────────────────
    stab_w = int(stability * 220)
    cv2.rectangle(frame, (12, h - bar_h + 60), (232, h - bar_h + 74),
                  (40, 40, 80), -1)
    cv2.rectangle(frame, (12, h - bar_h + 60),
                  (12 + stab_w, h - bar_h + 74), (100, 100, 255), -1)
    cv2.putText(frame, "Hold", (240, h - bar_h + 72),
                _FONT, 0.45, (160, 160, 200), 1, cv2.LINE_AA)

    # ── Cooldown flash ─────────────────────────────────────────────────
    if in_cooldown:
        cv2.putText(frame, "Registered!", (12, h - bar_h + 88),
                    _FONT, 0.5, _GREEN, 1, cv2.LINE_AA)

    # ── Status badge (top-right) ───────────────────────────────────────
    if not hand_tracking_available:
        badge_text  = "NO HAND MODEL"
        badge_color = (100, 100, 255)
    elif is_trained:
        badge_text  = "MODEL READY"
        badge_color = _GREEN
    else:
        badge_text  = "MODEL NOT TRAINED"
        badge_color = _ORANGE

    (tw, th), _ = cv2.getTextSize(badge_text, _FONT, 0.55, 1)
    cv2.rectangle(frame, (w - tw - 18, 8), (w - 4, 8 + th + 10), _DARK, -1)
    cv2.putText(frame, badge_text, (w - tw - 12, 8 + th + 4),
                _FONT, 0.55, badge_color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Landmark extraction (from MediaPipe Tasks result)
# ---------------------------------------------------------------------------

def _extract_landmarks_from_result(result) -> np.ndarray | None:
    """
    Extract a flat (63,) float32 array from a HandLandmarkerResult.
    Returns None if no hand was detected.
    """
    if not result.hand_landmarks:
        return None
    landmarks = result.hand_landmarks[0]  # first hand
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# Video streaming generator
# ---------------------------------------------------------------------------


def _generate_frames():
    """
    Generator that:
      1. Captures frames from the default camera.
      2. Runs MediaPipe hand-landmark detection (if model is available).
      3. Feeds landmarks into the sign model for prediction.
      4. Updates the global ``state``.
      5. Yields MJPEG-encoded frames.
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    landmarker = _try_init_hand_landmarker()
    hand_tracking_available = landmarker is not None
    timestamp_ms = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera read failed – retrying in 0.1 s")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # mirror

            current_sign = "NOTHING"
            confidence   = 0.0
            hand_detected = False

            if hand_tracking_available:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )
                timestamp_ms += 33  # ~30 fps

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    hand_detected = True
                    _draw_landmarks(frame, result.hand_landmarks[0])
                    raw_lm = _extract_landmarks_from_result(result)
                    if raw_lm is not None:
                        current_sign, confidence = sign_model.predict(raw_lm)

            # ── Update shared state ────────────────────────────────────
            with state.lock:
                state.hand_detected = hand_detected

                if state.cooldown_frames > 0:
                    state.cooldown_frames -= 1

                if current_sign == state.current_sign and current_sign != "NOTHING":
                    state.stability_count = min(
                        state.stability_count + 1, state.STABILITY_THRESHOLD
                    )
                else:
                    state.stability_count = 0

                state.current_sign = current_sign
                state.confidence   = confidence
                in_cooldown = state.cooldown_frames > 0
                stability   = state.stability_count / state.STABILITY_THRESHOLD

                # Accept sign when stability bar is full and not in cooldown
                if (
                    state.stability_count >= state.STABILITY_THRESHOLD
                    and state.cooldown_frames == 0
                    and sign_model.is_trained
                    and current_sign != "NOTHING"
                ):
                    if current_sign == "SPACE":
                        state.typed_text += " "
                    elif current_sign == "DELETE":
                        state.typed_text = state.typed_text[:-1]
                    else:
                        state.typed_text += current_sign

                    state.stability_count = 0
                    state.cooldown_frames = state.COOLDOWN_THRESHOLD

            # ── Draw HUD overlay ───────────────────────────────────────
            _draw_overlay(
                frame,
                current_sign,
                confidence,
                stability,
                sign_model.is_trained,
                in_cooldown,
                hand_tracking_available,
            )

            # ── Encode as JPEG ─────────────────────────────────────────
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
    finally:
        if landmarker is not None:
            landmarker.close()
        cap.release()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/state")
def get_state():
    with state.lock:
        return jsonify(
            {
                "current_sign": state.current_sign,
                "confidence": round(state.confidence, 4),
                "typed_text": state.typed_text,
                "stability": round(
                    state.stability_count / state.STABILITY_THRESHOLD, 4
                ),
                "model_trained": sign_model.is_trained,
                "in_cooldown": state.cooldown_frames > 0,
                "hand_detected": state.hand_detected,
            }
        )


@app.route("/api/clear", methods=["POST"])
def clear_text():
    with state.lock:
        state.typed_text = ""
    return jsonify({"ok": True})


@app.route("/api/backspace", methods=["POST"])
def backspace():
    with state.lock:
        state.typed_text = state.typed_text[:-1]
        return jsonify({"typed_text": state.typed_text})


@app.route("/api/space", methods=["POST"])
def add_space():
    with state.lock:
        state.typed_text += " "
        return jsonify({"typed_text": state.typed_text})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
