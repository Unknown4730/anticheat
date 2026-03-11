# Anti-Cheat Surveillance System

A real-time exam-monitoring application that uses **YOLOv8** object detection and **BoT-SORT** multi-object tracking to identify cheating behaviour in a live webcam feed.

---

## Features

- Real-time detection of `students_cheating` and `students_not_cheating` classes.
- Multi-object tracking with persistent student IDs (BoT-SORT).
- Strike-based alert system — raises a visual warning after 3 consecutive cheating detections for the same student.
- SQLite logging of all cheating events (auto-created on first run).
- Tkinter GUI with webcam selector, start/stop, and log-reset controls.
- Configurable model path via `--model` CLI flag or `MODEL_PATH` environment variable.

---

## Repository Layout

```
anticheat/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── gui.py          # Main GUI entrypoint
│   └── final3.py       # Headless (CLI) detection script
├── config/
│   ├── botsort.yaml    # BoT-SORT tracker settings
│   └── data.yaml       # Dataset definition for training
├── data/
│   ├── train/          # Training images & labels
│   ├── valid/          # Validation images & labels
│   └── test/           # Test images & labels
├── models/
│   └── yolov8n.pt      # Model weights (YOLOv8n base or fine-tuned)
└── docs/
    ├── usage.md
    └── development.md
```

> **Note:** `data/cheat_logs.db` is created automatically at runtime and is excluded from version control via `.gitignore`.

---

## Prerequisites

| Requirement | Recommended version |
|---|---|
| Python | 3.9 – 3.11 |
| PyTorch | ≥ 2.0 (CPU or CUDA) |
| CUDA (optional) | 11.8 or 12.x for GPU inference |
| Tkinter | Bundled with CPython; install `python3-tk` on Linux |

---

## Quickstart

### 1 – Clone and create a virtual environment

```bash
git clone https://github.com/Unknown4730/anticheat.git
cd anticheat
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or: .venv\Scripts\activate     # Windows
```

### 2 – Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU support, install the matching `torch+cu*` wheel **before** running pip install.  
> See <https://pytorch.org/get-started/locally/> for instructions.

### 3 – Ensure the model file is present

The repository ships with `models/yolov8n.pt` (the standard YOLOv8-nano base weights).  
To use your own fine-tuned weights, place them in `models/` and pass the path at startup (see below).

### 4 – Run the GUI

```bash
python src/gui.py
```

The application will:
1. Load `models/yolov8n.pt` by default.
2. Create `data/cheat_logs.db` automatically on first run.
3. Open a webcam-selector window; click **Start Detection** to begin.

#### Specifying a different model

```bash
# via CLI flag
python src/gui.py --model models/my_custom_weights.pt

# via environment variable
MODEL_PATH=models/my_custom_weights.pt python src/gui.py
```

### 5 – Run the headless (CLI) script

```bash
python src/final3.py
```

Press `q` to quit, `r` to reset logs.

---

## Dataset Layout

The dataset follows the [Roboflow](https://roboflow.com/) / YOLO format:

```
data/
├── train/
│   ├── images/   # .jpg training images
│   └── labels/   # YOLO .txt annotations
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Class definitions (see `config/data.yaml`):

| Index | Name |
|---|---|
| 0 | `person` |
| 1 | `students_cheating` |
| 2 | `students_not_cheating` |

Dataset source: [Roboflow – Exam Cheating](https://universe.roboflow.com/rtjhx/exam-cheating-jnmv1-hodrg/dataset/1) (CC BY 4.0).

---

## Retraining / Swapping Model Weights

See [docs/development.md](docs/development.md) for step-by-step retraining instructions.

Quick command:

```bash
yolo detect train \
    data=config/data.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640
```

After training, copy the best weights to `models/`:

```bash
cp runs/detect/train/weights/best.pt models/best.pt
python src/gui.py --model models/best.pt
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model file not found` error | Check that `models/yolov8n.pt` exists; use `--model` to point to a different file. |
| `ultralytics` import error | Run `pip install -r requirements.txt` inside your virtual environment. |
| Blank webcam / `Could not open video source` | Try index `1` or `2` in the webcam dropdown; ensure no other app is using the camera. |
| Tkinter not found (Linux) | `sudo apt-get install python3-tk` |
| CUDA / GPU errors | Install the CUDA-enabled PyTorch wheel matching your CUDA version. |
| Detections say "person" instead of cheating classes | You are using the base `yolov8n.pt`; retrain on the exam-cheating dataset and update `MODEL_PATH`. |

---

## License

This project is released under the [MIT License](LICENSE).
