# 🤟 Sign Language Translator

A real-time sign-language-to-text application that lets deaf users communicate with hearing people by simply showing ASL hand signs in front of a webcam.

![Sign Language Translator UI](https://github.com/user-attachments/assets/dc112679-2dec-4512-817d-c5165f947fc1)

---

## Features

| Feature | Details |
|---|---|
| 📷 Live hand tracking | MediaPipe detects all 21 hand landmarks in real time |
| 🧠 Sign recognition | TensorFlow/Keras MLP (falls back to scikit-learn RandomForest) |
| 💬 Auto-typing | Stable sign held for ~1 second is automatically typed |
| 🔊 Text-to-speech | Browser TTS reads your composed message aloud |
| 🎨 Modern dark UI | Glassmorphism design, confidence meters, stability bar |
| ⌨️ Manual controls | Space / Backspace / Clear / Copy buttons + keyboard shortcuts |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+** is required. A virtual environment is recommended.

### 2. Download the MediaPipe hand-tracking model

```bash
python scripts/download_models.py
```

This downloads `models/hand_landmarker.task` (~8 MB) from Google MediaPipe CDN.

### 3. Run the app (untrained model – hand tracking only)

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

The camera feed shows live hand-landmark tracking. Sign recognition will show
*"Model not trained"* until you complete steps 4–5 below.

---

## Training the Model

### Step 4 – Collect training data

```bash
python scripts/collect_data.py
```

A window opens showing your webcam feed.

| Key | Action |
|---|---|
| **C** | Capture current frame's hand landmarks |
| **N** / **Space** | Move to next sign class |
| **P** | Move to previous sign class |
| **Backspace** | Delete last captured sample |
| **Q** / **Esc** | Save and quit |

Aim for **≥ 100 samples per class**. Samples are saved to `data/<CLASS>.npy`.

### Step 5 – Train

```bash
python scripts/train_model.py
```

Optional flags:

```bash
python scripts/train_model.py --epochs 80 --batch-size 64
```

The trained model is saved to `model/saved_model/`.

### Step 6 – Restart the server

```bash
python app.py
```

The server loads the saved model automatically. The UI badge changes to
**"Model ready"** and signs are recognised in real time.

---

## Keyboard Shortcuts (in the browser)

| Key | Action |
|---|---|
| **Space** | Add a space |
| **Backspace** | Delete last character |
| **Escape** | Clear all text |
| **Enter** | Speak the current text |

---

## Project Structure

```
project-2026/
├── app.py                        # Flask application (video stream + REST API)
├── requirements.txt              # Python dependencies
├── models/                       # MediaPipe model bundles (downloaded separately)
├── model/
│   ├── __init__.py
│   └── sign_language_model.py    # Model architecture, training helpers & I/O
├── scripts/
│   ├── download_models.py        # Download MediaPipe hand landmarker model
│   ├── collect_data.py           # Interactive data collection tool
│   └── train_model.py            # Training & evaluation script
├── data/                         # Training data (.npy files, git-ignored)
├── static/
│   ├── css/style.css             # UI stylesheet
│   └── js/main.js                # Frontend JavaScript
└── templates/
    └── index.html                # Main HTML page
```

---

## Supported Signs (ASL alphabet)

A B C D E F G H I J K L M N O P Q R S T U V W X Y Z + **SPACE** + **DELETE**

---

## Tech Stack

- **Backend** – Python, Flask, OpenCV, MediaPipe
- **Model** – TensorFlow / Keras (MLP) with scikit-learn fallback
- **Frontend** – Vanilla HTML/CSS/JavaScript, Google Fonts (Inter)
- **TTS** – Web Speech API (browser-native, no external service)

---

## Contributing

Training data contributions are welcome! Collect samples for under-represented
signs or accents and open a pull request with the updated `.npy` files.

---

*Built with ❤️ to bridge communication between the deaf and hearing communities.*
