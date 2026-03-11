# Development Guide

This guide covers the project architecture, model training, dataset details, and instructions for contributors.

---

## Architecture Overview

```
Webcam (OpenCV)
    │
    ▼
YOLOv8 model.track()          ← loads models/best.pt
    │
    ├─ Bounding boxes + class IDs + confidence scores
    │
    ▼
BoT-SORT tracker (config/botsort.yaml)
    │  Assigns stable per-student IDs across frames
    │
    ▼
Strike counter (in-memory defaultdict)
    │
    ├─ Strike ≤ MAX_STRIKES  →  log to SQLite (cheat_logs.db)
    └─ Strike ≥ MAX_STRIKES  →  overlay "WARNING" on video frame
    │
    ▼
Tkinter GUI + OpenCV imshow window
```

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| GUI & detection loop | `src/gui.py` | Main application, Tkinter UI, background detection thread |
| Trained weights | `models/best.pt` | Custom YOLOv8 model trained on the exam cheating dataset |
| Base weights | `models/yolov8n.pt` | Pretrained YOLOv8-nano (used as the starting checkpoint for fine-tuning) |
| Dataset | `data/` | Images + YOLO-format labels for train / valid / test splits |
| Dataset config | `data/data.yaml` | Class names and split paths for YOLOv8 training |
| Tracker config | `config/botsort.yaml` | BoT-SORT hyperparameters (thresholds, buffer sizes, etc.) |

---

## Dataset Details

**Source:** [Roboflow Universe – exam-cheating](https://universe.roboflow.com/rtjhx/exam-cheating-jnmv1-hodrg/dataset/1)
**License:** CC BY 4.0

### Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | `person` | Generic person (unlabelled behaviour) |
| 1 | `students_cheating` | Student showing cheating behaviour |
| 2 | `students_not_cheating` | Student behaving normally |

### Splits

| Split | Location |
|-------|---------|
| Train | `data/train/` |
| Validation | `data/valid/` |
| Test | `data/test/` |

Labels are in YOLO format: one `.txt` file per image with rows of `class cx cy w h` (normalised coordinates).

---

## Re-training the Model

### 1. Prepare the environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train from the pretrained base

```bash
yolo detect train \
    model=models/yolov8n.pt \
    data=data/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=0            # GPU index; use 'cpu' for CPU-only
```

The best weights will be saved to `runs/detect/train/weights/best.pt`. Copy them to `models/best.pt` when training is complete:

```bash
cp runs/detect/train/weights/best.pt models/best.pt
```

### 3. Evaluate

```bash
yolo detect val \
    model=models/best.pt \
    data=data/data.yaml \
    imgsz=640
```

### Training Tips

- Increase `epochs` (100–200) for better convergence if you have enough data.
- Use `augment=True` or tune `hsv_*`, `fliplr`, `mosaic` in your data config.
- Use `device=0` (GPU) for practical training times. CPU training on this dataset takes several hours.
- Monitor `runs/detect/train/results.png` for loss and mAP curves.

---

## BoT-SORT Tracker Configuration

The tracker is configured in `config/botsort.yaml`. Key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `track_high_thresh` | 0.6 | Detections above this are considered "high confidence" |
| `track_low_thresh` | 0.1 | Minimum confidence to associate to an existing track |
| `new_track_thresh` | 0.7 | Minimum confidence to start a new track |
| `track_buffer` | 30 | Frames to keep a lost track before removing it |
| `match_thresh` | 0.8 | IoU threshold for track/detection association |
| `with_reid` | False | Disable Re-ID (set `True` for better multi-camera tracking) |

---

## Code Style & Contribution Guidelines

### Setting Up for Development

```bash
git clone https://github.com/Unknown4730/anticheat.git
cd anticheat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Use descriptive variable and function names.
- Keep functions small and single-purpose.
- All user-facing strings should be clear and informative.

### Branch Naming

| Purpose | Pattern | Example |
|---------|---------|---------|
| New feature | `feature/<name>` | `feature/export-csv-logs` |
| Bug fix | `fix/<name>` | `fix/camera-index-crash` |
| Documentation | `docs/<name>` | `docs/update-usage-guide` |

### Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure the application starts and runs without errors.
4. Open a pull request with a clear description of what you changed and why.

---

## Project Conventions

- **No hardcoded paths.** All file paths must be derived from `__file__`, passed via CLI argument, or read from environment variables.
- **No committed runtime artifacts.** Database files (`.db`), training run outputs (`runs/`), and Python bytecode (`__pycache__/`, `*.pyc`) are excluded by `.gitignore`.
- **Single entry point.** `src/gui.py` is the only supported entry point for end users.
- **Model weights via Git LFS.** Large binary files (`*.pt`) are tracked with Git LFS. Ensure `git lfs` is installed before cloning or pushing model changes.

---

## Git LFS

This repository uses [Git Large File Storage](https://git-lfs.github.com/) for `.pt` model weight files.

To clone with model files:

```bash
git lfs install
git clone https://github.com/Unknown4730/anticheat.git
```

If you already cloned without LFS:

```bash
git lfs pull
```
