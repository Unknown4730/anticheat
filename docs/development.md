# Development Guide

## Project Structure

```
anticheat/
├── src/
│   ├── gui.py        # GUI entrypoint — webcam selector + Tkinter window
│   └── final3.py     # Headless detection loop (no GUI)
├── config/
│   ├── botsort.yaml  # BoT-SORT tracker hyperparameters
│   └── data.yaml     # Dataset paths and class definitions
├── data/
│   ├── train/        # Training split (images + labels)
│   ├── valid/        # Validation split
│   └── test/         # Test split
├── models/
│   └── yolov8n.pt    # Model weights
├── docs/
│   ├── usage.md      # End-user usage guide
│   └── development.md  # (this file)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Setting Up a Development Environment

```bash
git clone https://github.com/Unknown4730/anticheat.git
cd anticheat
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Retraining the Model

### 1 – Verify the dataset

Check that your images and YOLO-format `.txt` label files are in:

```
data/train/images/   data/train/labels/
data/valid/images/   data/valid/labels/
data/test/images/    data/test/labels/
```

Each `.txt` file should contain one detection per line:  
`<class_index> <x_center> <y_center> <width> <height>` (all normalised 0–1).

Class indices (defined in `config/data.yaml`):

| 0 | 1 | 2 |
|---|---|---|
| `person` | `students_cheating` | `students_not_cheating` |

### 2 – Run training

```bash
yolo detect train \
    data=config/data.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    name=anticheat_v1
```

Training artefacts are written to `runs/detect/anticheat_v1/`.

### 3 – Evaluate

```bash
yolo detect val \
    data=config/data.yaml \
    model=runs/detect/anticheat_v1/weights/best.pt
```

### 4 – Deploy

Copy the best weights to `models/` and update your run command:

```bash
cp runs/detect/anticheat_v1/weights/best.pt models/anticheat_v1.pt
python src/gui.py --model models/anticheat_v1.pt
```

---

## Adjusting Detection Parameters

Edit the constants near the top of `src/gui.py` (or `src/final3.py`):

| Constant | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence to process a box |
| `MAX_STRIKES` | `3` | Frames of cheating before the warning overlay appears |

---

## Tracker Configuration

`config/botsort.yaml` exposes the key BoT-SORT knobs:

| Parameter | Effect |
|---|---|
| `track_high_thresh` | Confidence floor for starting a track |
| `track_buffer` | Frames to keep a "lost" track alive |
| `match_thresh` | IoU threshold for matching detections to tracks |
| `with_reid` | Enable appearance-based Re-ID (set `True` for better accuracy but higher CPU/GPU use) |

---

## Adding a New Class

1. Collect and label images with the new class using [Roboflow](https://roboflow.com/) or [Label Studio](https://labelstud.io/).
2. Export in YOLO format and merge with the existing `data/` splits.
3. Update `config/data.yaml`:
   - Increment `nc`.
   - Append the new class name to `names`.
4. Retrain as above.
5. Update any hard-coded class-name comparisons in `src/gui.py` / `src/final3.py` if needed.

---

## Code Style

The project uses standard Python 3 conventions.  
Run a quick lint check with:

```bash
pip install flake8
flake8 src/ --max-line-length=100
```

---

## Git LFS for Large Model Files

If a weights file exceeds 50 MB, enable Git LFS to avoid bloating the repository:

```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git commit -m "Track model weights with LFS"
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-improvement`.
3. Commit your changes with clear messages.
4. Open a pull request against `main`.
