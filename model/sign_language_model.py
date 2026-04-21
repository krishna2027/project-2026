"""
Sign Language Recognition Model

Architecture:
  - Input: 63 features (21 MediaPipe hand landmarks × 3 coordinates: x, y, z)
  - Hidden layers: Dense(256) → Dense(128) → Dense(64) with BatchNorm & Dropout
  - Output: 29 ASL classes (A–Z + SPACE + DELETE + NOTHING)

The user trains this model by:
  1. Running scripts/collect_data.py to capture landmark data per sign.
  2. Running scripts/train_model.py to fit and save the model.
"""

import os
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Try importing TensorFlow; fall back to scikit-learn if unavailable
try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available – falling back to scikit-learn.")

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE", "NOTHING"]
NUM_CLASSES = len(ASL_CLASSES)  # 29
INPUT_SIZE = 63  # 21 landmarks × 3 (x, y, z)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.keras")
SKLEARN_MODEL_PATH = os.path.join(MODEL_DIR, "sign_model_sklearn.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")


# ---------------------------------------------------------------------------
# Landmark normalisation helper
# ---------------------------------------------------------------------------

def normalise_landmarks(raw: np.ndarray) -> np.ndarray:
    """
    Normalise 63-element landmark vector so that the hand is translation-
    and scale-invariant.

    Steps:
      1. Reshape to (21, 3) – each row is (x, y, z) for one landmark.
      2. Translate so that the wrist (landmark 0) is at the origin.
      3. Scale so that the maximum Euclidean distance from the wrist is 1.
    """
    pts = raw.reshape(21, 3).copy()
    pts -= pts[0]  # translate
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


# ---------------------------------------------------------------------------
# Keras (TF) model
# ---------------------------------------------------------------------------

def _build_keras_model(num_classes: int) -> "keras.Model":
    """Return a compiled Keras model for landmark-based sign classification."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(INPUT_SIZE,), name="landmarks"),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="sign_language_model",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Main model wrapper
# ---------------------------------------------------------------------------

class SignLanguageModel:
    """
    Unified wrapper around the sign language recognition model.

    Provides a backend-agnostic ``predict`` API.  When TensorFlow is
    available it uses a Keras MLP; otherwise it falls back to a scikit-learn
    RandomForestClassifier.
    """

    def __init__(self):
        self.classes: list[str] = ASL_CLASSES
        self.num_classes: int = NUM_CLASSES
        self.is_trained: bool = False
        self._keras_model = None
        self._sklearn_model = None
        self._backend: str = "none"

        # Try to build (and possibly load) the model
        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        if TF_AVAILABLE:
            self._keras_model = _build_keras_model(self.num_classes)
            self._backend = "keras"
            if os.path.exists(KERAS_MODEL_PATH):
                try:
                    self._keras_model = keras.models.load_model(KERAS_MODEL_PATH)
                    self.is_trained = True
                    self._load_metadata()
                    logger.info("Loaded trained Keras model from %s", KERAS_MODEL_PATH)
                except Exception as exc:
                    logger.error("Could not load Keras model: %s", exc)
        elif SKLEARN_AVAILABLE:
            self._backend = "sklearn"
            if os.path.exists(SKLEARN_MODEL_PATH):
                try:
                    self._sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
                    self.is_trained = True
                    self._load_metadata()
                    logger.info(
                        "Loaded trained sklearn model from %s", SKLEARN_MODEL_PATH
                    )
                except Exception as exc:
                    logger.error("Could not load sklearn model: %s", exc)
        else:
            logger.warning("No ML backend available (install TensorFlow or scikit-learn).")

    def _load_metadata(self) -> None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as fh:
                meta = json.load(fh)
            self.classes = meta.get("classes", self.classes)
            self.num_classes = len(self.classes)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, raw_landmarks: np.ndarray) -> tuple[str, float]:
        """
        Predict the sign from raw MediaPipe hand landmarks.

        Args:
            raw_landmarks: 1-D numpy array of length 63.

        Returns:
            (class_name, confidence) – confidence is in [0, 1].
            Returns ('NOTHING', 0.0) when the model has not been trained.
        """
        if not self.is_trained:
            return "NOTHING", 0.0

        landmarks = normalise_landmarks(np.asarray(raw_landmarks, dtype=np.float32))

        if self._backend == "keras" and self._keras_model is not None:
            probs = self._keras_model.predict(
                landmarks.reshape(1, -1), verbose=0
            )[0]
            idx = int(np.argmax(probs))
            return self.classes[idx], float(probs[idx])

        if self._backend == "sklearn" and self._sklearn_model is not None:
            probs = self._sklearn_model.predict_proba(landmarks.reshape(1, -1))[0]
            idx = int(np.argmax(probs))
            return self.classes[idx], float(probs[idx])

        return "NOTHING", 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save the trained model and class metadata to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        if self._backend == "keras" and self._keras_model is not None:
            self._keras_model.save(KERAS_MODEL_PATH)
            logger.info("Saved Keras model to %s", KERAS_MODEL_PATH)
        elif self._backend == "sklearn" and self._sklearn_model is not None:
            joblib.dump(self._sklearn_model, SKLEARN_MODEL_PATH)
            logger.info("Saved sklearn model to %s", SKLEARN_MODEL_PATH)

        meta = {"classes": self.classes, "backend": self._backend}
        with open(METADATA_PATH, "w") as fh:
            json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------
    # Training helpers (used by scripts/train_model.py)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.15,
    ) -> dict:
        """
        Train the model.

        Args:
            X: Feature matrix of shape (N, 63) – raw landmarks.
            y: Integer label array of shape (N,).
            epochs: Training epochs (Keras only).
            batch_size: Mini-batch size (Keras only).
            validation_split: Fraction of data held out for validation.

        Returns:
            dict with at least 'accuracy' key (final training accuracy).
        """
        # Normalise every sample
        X_norm = np.array([normalise_landmarks(row) for row in X])

        if self._backend == "keras" and TF_AVAILABLE:
            # Rebuild with correct number of classes if needed
            actual_classes = sorted(set(int(v) for v in y))
            num_classes = max(actual_classes) + 1
            if num_classes != self.num_classes:
                self.num_classes = num_classes
                self._keras_model = _build_keras_model(num_classes)

            history = self._keras_model.fit(
                X_norm,
                y.astype(np.int32),
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
            )
            self.is_trained = True
            acc = float(history.history["accuracy"][-1])
            return {"accuracy": acc, "history": history.history}

        if self._backend == "sklearn" and SKLEARN_AVAILABLE:
            self._sklearn_model = RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )
            self._sklearn_model.fit(X_norm, y.astype(int))
            self.is_trained = True
            acc = float(self._sklearn_model.score(X_norm, y.astype(int)))
            return {"accuracy": acc}

        raise RuntimeError("No ML backend available for training.")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"SignLanguageModel(backend={self._backend!r}, status={status!r})"
