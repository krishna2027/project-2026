"""
Model Training Script for Sign Language Translator
====================================================

Usage
-----
    python scripts/train_model.py [--epochs N] [--batch-size N]

What it does
-------------
1. Loads all .npy landmark files from data/.
2. Builds the class list from the files present.
3. Trains the SignLanguageModel (TF Keras preferred, else scikit-learn).
4. Saves the trained model to model/saved_model/.
5. Prints final accuracy.

Training data format
---------------------
Each data/<CLASS_NAME>.npy file must be a float32 array of shape (N, 63)
produced by scripts/collect_data.py.
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.sign_language_model import (  # noqa: E402
    ASL_CLASSES,
    SignLanguageModel,
)

DATA_DIR = os.path.join(_ROOT, "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all .npy files in *data_dir* and build (X, y, class_names).

    Returns
    -------
    X           : float32 array of shape (total_samples, 63)
    y           : int32 array of shape (total_samples,)
    class_names : list of class name strings corresponding to label indices
    """
    files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".npy")
    )
    if not files:
        raise FileNotFoundError(
            f"No .npy files found in {data_dir}. "
            "Run  python scripts/collect_data.py  first."
        )

    # Build a class-name list from the files that exist; preserve canonical
    # order where possible.
    present = [os.path.splitext(f)[0] for f in files]
    ordered = [c for c in ASL_CLASSES if c in present] + [
        c for c in present if c not in ASL_CLASSES
    ]

    label_map = {cls: idx for idx, cls in enumerate(ordered)}

    all_X: list[np.ndarray] = []
    all_y: list[int]        = []

    for cls in ordered:
        path    = os.path.join(data_dir, f"{cls}.npy")
        samples = np.load(path, allow_pickle=False).astype(np.float32)
        n       = len(samples)
        if n == 0:
            print(f"  [skip] {cls}  – 0 samples")
            continue
        print(f"  [load] {cls:8s}  {n:4d} samples")
        all_X.append(samples)
        all_y.extend([label_map[cls]] * n)

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y, dtype=np.int32)
    return X, y, ordered


def print_report(class_names: list[str], y_true, y_pred) -> None:
    """Print a per-class accuracy table."""
    from collections import defaultdict

    correct = defaultdict(int)
    total   = defaultdict(int)
    for yt, yp in zip(y_true, y_pred):
        total[yt]   += 1
        correct[yt] += int(yt == yp)

    print("\n── Per-class accuracy ──────────────────────────")
    for idx, cls in enumerate(class_names):
        if total[idx]:
            pct = 100.0 * correct[idx] / total[idx]
            bar = "█" * int(pct / 5)
            print(f"  {cls:8s}  {pct:5.1f}%  {bar}")
    print("────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the sign language model.")
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Training epochs (TF only, default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (TF only, default: 32)")
    args = parser.parse_args()

    print("\n===  Sign Language Model Trainer  ===\n")
    print(f"Loading data from {DATA_DIR} …")

    X, y, class_names = load_dataset(DATA_DIR)
    print(f"\nLoaded {len(X)} samples across {len(class_names)} classes.\n")

    model = SignLanguageModel()
    # Override the class list so it matches the actual training data
    model.classes      = class_names
    model.num_classes  = len(class_names)
    # Rebuild Keras model head if needed
    try:
        import tensorflow as tf  # noqa: F401
        from model.sign_language_model import _build_keras_model
        model._keras_model = _build_keras_model(model.num_classes)
    except ImportError:
        pass

    print("Training …\n")
    result = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    final_acc = result.get("accuracy", 0.0)
    print(f"\nFinal training accuracy: {final_acc:.2%}")

    # Validation accuracy (simple split)
    try:
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        val_model = SignLanguageModel()
        val_model.classes     = class_names
        val_model.num_classes = len(class_names)
        try:
            from model.sign_language_model import _build_keras_model
            val_model._keras_model = _build_keras_model(val_model.num_classes)
        except ImportError:
            pass
        val_model.fit(X_train, y_train, epochs=args.epochs,
                      batch_size=args.batch_size)

        # Quick per-class report
        from model.sign_language_model import normalise_landmarks
        X_val_norm = np.array([normalise_landmarks(r) for r in X_val])
        if val_model._backend == "keras":
            probs  = val_model._keras_model.predict(X_val_norm, verbose=0)
            y_pred = probs.argmax(axis=1)
        else:
            y_pred = val_model._sklearn_model.predict(X_val_norm)
        print_report(class_names, y_val, y_pred)
    except Exception as exc:
        print(f"  (Skipped validation report: {exc})")

    print("\nSaving model …")
    model.save()
    print(f"  Saved to {os.path.join(_ROOT, 'model', 'saved_model')}")
    print("\nRestart the Flask server to load the trained model.\n")


if __name__ == "__main__":
    main()
