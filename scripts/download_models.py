"""
Download required MediaPipe model bundles
==========================================

Usage
-----
    python scripts/download_models.py

Downloads
---------
- models/hand_landmarker.task
  Used by the Flask app (app.py) for real-time hand landmark detection.
"""

import os
import sys
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "models")

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def download(url: str, dest: str) -> None:
    filename = os.path.basename(dest)
    print(f"  Downloading {filename} …", end=" ", flush=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 // total_size, 100)
            print(f"\r  Downloading {filename} … {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  ✓ {filename} saved to {dest}")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("\n===  MediaPipe Model Downloader  ===\n")

    any_downloaded = False
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"  ✓ {filename} already present, skipping.")
            continue
        try:
            download(url, dest)
            any_downloaded = True
        except Exception as exc:
            print(f"\n  ✗ Failed to download {filename}: {exc}")
            print(
                f"  Please download it manually from:\n    {url}\n"
                f"  and place it at:\n    {dest}\n"
            )
            sys.exit(1)

    if any_downloaded:
        print("\nAll models downloaded. You can now start the app:")
        print("    python app.py\n")
    else:
        print("\nAll models already present. Start the app with:")
        print("    python app.py\n")


if __name__ == "__main__":
    main()
