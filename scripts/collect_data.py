"""
Data Collection Script for Sign Language Translator
====================================================

Usage
-----
    python scripts/collect_data.py

Prerequisites
-------------
Run this first to get the MediaPipe hand landmarker model:

    python scripts/download_models.py

Controls (while the window is open)
------------------------------------
    C / c       – Capture the current frame's hand landmarks
    N / n       – Move to the next class (letter/sign)
    P / p       – Go back to the previous class
    Q / q       – Quit and save collected data
    SPACE       – Same as N (next)
    BACKSPACE   – Delete the last captured sample for current class

What it does
-------------
For each ASL class (A–Z, SPACE, DELETE, NOTHING):
  1. Shows the camera with hand-landmark overlay.
  2. Prompts you to sign that letter.
  3. Press C to capture a sample; aim for ≥ 100 samples per class.
  4. Repeats for every class, then saves numpy arrays to data/.

Output
------
  data/<CLASS_NAME>.npy   – shape (N, 63) float32 array of raw landmarks
"""

import os
import sys
import time

import cv2
import numpy as np

# Ensure the project root is on sys.path so we can import model utilities
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.sign_language_model import ASL_CLASSES  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR          = os.path.join(_ROOT, "data")
MODELS_DIR        = os.path.join(_ROOT, "models")
HAND_MODEL_PATH   = os.path.join(MODELS_DIR, "hand_landmarker.task")
SAMPLES_PER_CLASS = 150           # target samples per class
CAM_INDEX         = 0

# Hand connection pairs (MediaPipe landmark indices)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_CYAN   = (0, 212, 255)
_GREEN  = (0, 255, 150)
_RED    = (0, 80, 255)
_WHITE  = (255, 255, 255)
_DARK   = (20, 20, 40)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing(class_name: str) -> list[np.ndarray]:
    path = os.path.join(DATA_DIR, f"{class_name}.npy")
    if os.path.exists(path):
        arr = np.load(path, allow_pickle=False)
        return list(arr)
    return []


def _save(class_name: str, samples: list[np.ndarray]) -> None:
    if not samples:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{class_name}.npy")
    np.save(path, np.array(samples, dtype=np.float32))


def _extract_landmarks(result) -> np.ndarray | None:
    if not result.hand_landmarks:
        return None
    coords = []
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def _draw_hand(frame: np.ndarray, landmarks: list) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 120), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 5 if i == 0 else 4
        cv2.circle(frame, (x, y), r, _CYAN, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r + 1, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_ui(frame, class_name, class_idx, total_classes,
             n_captured, target, countdown, hand_ok):
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 70), _DARK, -1)

    cv2.putText(frame, f"Class {class_idx + 1}/{total_classes}:  {class_name}",
                (12, 28), _FONT, 0.85, _CYAN, 2, cv2.LINE_AA)

    progress = min(n_captured / target, 1.0)
    bar_w = w - 24
    cv2.rectangle(frame, (12, 40), (12 + bar_w, 58), (50, 50, 70), -1)
    fill_color = _GREEN if progress >= 1.0 else (100, 180, 255)
    cv2.rectangle(frame, (12, 40),
                  (12 + int(bar_w * progress), 58), fill_color, -1)
    cv2.putText(frame, f"{n_captured}/{target}", (12 + bar_w + 6, 55),
                _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)

    # Bottom controls
    cv2.rectangle(frame, (0, h - 50), (w, h), _DARK, -1)
    controls = "[C] Capture   [N] Next   [P] Prev   [Bksp] Undo   [Q] Quit"
    cv2.putText(frame, controls, (10, h - 18), _FONT, 0.5, (160, 160, 200),
                1, cv2.LINE_AA)

    # Countdown after capture
    if countdown > 0:
        cv2.putText(frame, f"Captured! ({n_captured})", (12, h - 65),
                    _FONT, 0.7, _GREEN, 2, cv2.LINE_AA)

    # No-hand warning
    if not hand_ok:
        cv2.putText(frame, "No hand detected", (12, h - 65),
                    _FONT, 0.7, _RED, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(HAND_MODEL_PATH):
        print(
            f"\nERROR: Hand landmarker model not found at {HAND_MODEL_PATH}\n"
            "Run  python scripts/download_models.py  first.\n"
        )
        sys.exit(1)

    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base

    BaseOptions = mp_base.BaseOptions
    HandLandmarker = mp_vision.HandLandmarker
    HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
    RunningMode = mp_vision.RunningMode

    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    classes     = ASL_CLASSES
    class_idx   = 0
    countdown   = 0   # frames to show "Captured!" message
    timestamp_ms = 0

    # Load any previously saved data
    collected: dict[str, list[np.ndarray]] = {
        cls: _load_existing(cls) for cls in classes
    }

    print("\n===  Sign Language Data Collector  ===")
    print(f"  Target: {SAMPLES_PER_CLASS} samples per class")
    print("  Controls: [C] Capture   [N] Next   [P] Prev   [Q] Quit\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                timestamp_ms += 33

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, timestamp_ms)

                raw_lm  = _extract_landmarks(result)
                hand_ok = raw_lm is not None

                if result.hand_landmarks:
                    _draw_hand(frame, result.hand_landmarks[0])

                cls_name   = classes[class_idx]
                n_captured = len(collected[cls_name])

                if countdown > 0:
                    countdown -= 1

                _draw_ui(frame, cls_name, class_idx, len(classes),
                         n_captured, SAMPLES_PER_CLASS, countdown, hand_ok)

                cv2.imshow("Sign Language Data Collector", frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                elif key in (ord("c"), ord("C")):
                    if raw_lm is not None:
                        collected[cls_name].append(raw_lm)
                        _save(cls_name, collected[cls_name])
                        countdown = 20
                        print(f"  [{cls_name}] {len(collected[cls_name])}/{SAMPLES_PER_CLASS} samples")

                elif key in (ord("n"), ord("N"), ord(" ")):
                    _save(cls_name, collected[cls_name])
                    class_idx = (class_idx + 1) % len(classes)

                elif key in (ord("p"), ord("P")):
                    _save(cls_name, collected[cls_name])
                    class_idx = (class_idx - 1) % len(classes)

                elif key == 8:  # Backspace – undo last capture
                    if collected[cls_name]:
                        collected[cls_name].pop()
                        _save(cls_name, collected[cls_name])
                        print(f"  [{cls_name}] Removed 1 → {len(collected[cls_name])}")

        finally:
            # Save everything on exit
            for cls_name, samples in collected.items():
                _save(cls_name, samples)

            cap.release()
            cv2.destroyAllWindows()

    print("\nData saved to data/")
    totals = {cls: len(collected[cls]) for cls in classes if collected[cls]}
    for cls, n in totals.items():
        marker = "✓" if n >= SAMPLES_PER_CLASS else "·"
        print(f"  {marker} {cls:8s}  {n:4d} samples")
    print("\nRun  python scripts/train_model.py  to train the model.\n")


if __name__ == "__main__":
    main()

