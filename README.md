# Anticheat — Classroom Cheating Detection

A classroom monitoring tool that uses a YOLOv8-based detector and BoT-SORT tracker to identify students in a live video feed and flag suspected cheating. Each student is tracked individually, with a configurable **3-strike rule**: on the third confirmed cheating detection the student is permanently flagged. All detections are displayed with colour-coded bounding boxes and persisted to a SQLite audit log.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Running the GUI](#running-the-gui)
- [Dataset & Model](#dataset--model)
- [Database & Logs](#database--logs)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project detects students in a classroom-like video feed and classifies each detection as either **cheating** (red box) or **not cheating** (green box). Each student is assigned a persistent ID by the BoT-SORT tracker (with a centroid-based fallback), and strikes are counted per student. When a student reaches the configured strike limit (default **3**), they are permanently flagged as cheating and highlighted in red for the remainder of the session.

---

## Features

- **Multi-student detection** — designed for classroom environments with many simultaneous subjects.
- **Per-student 3-strike rule** — each student gets 3 chances; absolute cheating is only declared on the third strike.
- **Colour-coded bounding boxes** — green for OK, orange for warning (1–2 strikes), red for confirmed cheating (≥ 3 strikes).
- **BoT-SORT tracking** — stable per-student IDs across frames; centroid-based fallback when tracker IDs are unavailable.
- **Tkinter GUI** — side panel listing active students, strike counts, and statuses with Start / Stop / Reset controls.
- **CSV export** — one-click export of all flagged students.
- **Persistent audit log** — all cheating detections are written to a runtime SQLite database.
- **Fully configurable** — model path, thresholds, labels, and tracker settings are all adjustable via CLI flags or environment variables.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8 or newer |
| **GPU (recommended)** | NVIDIA GPU with CUDA 11.8+ for real-time inference. CPU-only inference is supported but will be significantly slower. |
| **PyTorch** | Install the CUDA-enabled build matching your CUDA version (see [pytorch.org/get-started](https://pytorch.org/get-started/locally/)). |
| **System libraries** | On Ubuntu/Debian, Tkinter may need: `sudo apt-get install python3-tk` |

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Unknown4730/anticheat.git
   cd anticheat
   ```

2. **(Optional) Pull model weights via Git LFS** if they were not included in the clone:

   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv

   # macOS / Linux
   source .venv/bin/activate

   # Windows PowerShell
   .venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```

4. **(Optional) Install a CUDA-enabled PyTorch** for GPU acceleration (replace `cu118` with your CUDA version):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

---

## Quickstart

Run the GUI using the default model at `models/best.pt` (custom trained weights):

```bash
python -m src.gui
```

Specify a different model path and override thresholds:

```bash
python -m src.gui --model ./models/yolov8n.pt --conf 0.5 --max-strikes 3
```

Use environment variables instead of CLI flags:

```bash
MODEL_PATH=./models/best.pt CONFIDENCE_THRESHOLD=0.6 MAX_STRIKES=3 python -m src.gui
```

Validate strike logic without a camera (headless simulation):

```bash
python simulate_detections.py
```

---

## Configuration

All settings can be overridden via **CLI arguments** (highest priority) or **environment variables** (fallback).

### CLI Arguments

```
python -m src.gui --help

  -m, --model PATH        Path to model weights  [default: $MODEL_PATH or ./models/yolov8n.pt]
      --conf FLOAT        Confidence threshold 0–1  [default: 0.5]
      --max-strikes INT   Strikes before a student is flagged  [default: 3]
      --tracker-config PATH  Tracker YAML config path  [default: config/botsort.yaml]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/yolov8n.pt` | Path to the YOLO model weights (`.pt` file). |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence (0–1). Raise to reduce false positives. |
| `MAX_STRIKES` | `3` | Number of cheating detections allowed before a student is permanently flagged. |
| `MOVEMENT_THRESHOLD_PIXELS` | `30` | Centroid movement (pixels) considered a notable position change for fallback tracking. |
| `LABEL_CHEATING` | `students_cheating` | Model class name for cheating detections. Change if your model uses different labels. |
| `LABEL_OK` | `students_not_cheating` | Model class name for normal (non-cheating) detections. |
| `TRACKER_CONFIG` | `config/botsort.yaml` | Path to the BoT-SORT tracker YAML configuration file. |

**Example — custom labels:**

```bash
LABEL_CHEATING=cheating LABEL_OK=ok python -m src.gui --model ./models/custom.pt
```

---

## Running the GUI

`src/gui.py` is the primary application entrypoint. On launch it:

1. Validates the model path and exits with a clear error if the file is missing.
2. Creates `runtime/cheat_logs.db` (and the `runtime/` directory) if they do not exist.
3. Opens the Tkinter control panel.

**Control panel buttons:**

| Button | Action |
|--------|--------|
| **Start** | Opens the selected camera and begins detection in a background thread. A separate OpenCV window shows the annotated live feed. Press `q` in that window to stop. |
| **Stop** | Stops the detection thread and releases the camera. |
| **Reset Logs** | Clears the `detections` table in the database and resets all in-memory strike counters. Use this at the start of each new exam session. |
| **Export Flagged CSV** | Saves a CSV file listing all students whose status is `cheating` (student ID and strike count). |

The **live feed window** overlays a bounding box on each detected student:

- 🟢 **Green** — not cheating (0 strikes)
- 🟠 **Orange** — warning (1–2 strikes)
- 🔴 **Red** — confirmed cheating (≥ 3 strikes)

---

## Dataset & Model

### Repository Structure

```
.
├── src/                    # Application source code
│   └── gui.py              # Main GUI and detection entrypoint
├── models/
│   ├── best.pt             # Custom-trained YOLOv8 weights (primary model)
│   └── yolov8n.pt          # Pretrained YOLOv8-nano base weights
├── data/
│   ├── data.yaml           # YOLOv8 dataset config (class names, split paths)
│   ├── train/              # Training images and YOLO-format labels
│   └── valid/              # Validation images and YOLO-format labels
├── config/
│   └── botsort.yaml        # BoT-SORT tracker configuration
├── runtime/                # Created at runtime; excluded from git
│   └── cheat_logs.db       # SQLite audit log
├── docs/
│   ├── usage.md            # Detailed usage guide
│   └── development.md      # Architecture, training, and contribution guide
├── simulate_detections.py  # Headless strike-logic simulation (no camera needed)
├── requirements.txt
├── LICENSE
└── README.md
```

### Dataset

The model was trained on the [Roboflow exam-cheating dataset](https://universe.roboflow.com/rtjhx/exam-cheating-jnmv1-hodrg/dataset/1) (CC BY 4.0). It has three classes:

| ID | Class name | Description |
|----|-----------|-------------|
| 0 | `person` | Generic person (unlabelled behaviour) |
| 1 | `students_cheating` | Student exhibiting cheating behaviour |
| 2 | `students_not_cheating` | Student behaving normally |

### Git LFS for Model Weights

Model weight files (`.pt`) are large binary files. If you plan to commit updated weights, use [Git LFS](https://git-lfs.github.com/):

```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git commit -m "track model weights with Git LFS"
```

---

## Database & Logs

Detection events are persisted to **`runtime/cheat_logs.db`** (created automatically at first run). The `detections` table schema:

```sql
CREATE TABLE detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,     -- tracker-assigned student ID
    timestamp  TEXT,     -- YYYY-MM-DD HH:MM:SS
    label      TEXT,     -- detected class name
    strikes    INTEGER   -- cumulative strike count at time of event
);
```

**Query the log directly:**

```bash
sqlite3 runtime/cheat_logs.db "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 20;"
```

**Reset Logs** (GUI button) deletes all rows from `detections` and clears in-memory counters — use this at the start of a new exam session.

The `runtime/` directory is listed in `.gitignore` so database files are never committed.

---

## Troubleshooting

### Model file not found

```
Model not found at ./models/yolov8n.pt
```

Confirm the weights file exists. If you have custom weights, pass the path explicitly:

```bash
python -m src.gui --model /path/to/your_weights.pt
```

If using Git LFS, pull the files first: `git lfs pull`

---

### `ultralytics` import error

```
ERROR: ultralytics module not found.
```

Ensure the virtual environment is active and dependencies are installed:

```bash
pip install -r requirements.txt
```

Check that the installed `ultralytics` and `torch` versions are compatible (see [ultralytics docs](https://docs.ultralytics.com)).

---

### Camera not found / cannot open camera

- Verify no other application is using the camera.
- On Linux, check permissions: `ls -l /dev/video*`
- Try changing the camera index in the **Camera** field of the control panel (0, 1, 2, …).

---

### Tkinter not available (Linux)

```bash
sudo apt-get install python3-tk
```

---

### Very slow detection / low FPS

Install the CUDA-enabled PyTorch build matching your GPU's CUDA version:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Wrong labels / unexpected class names

The application expects detections labelled `students_cheating` and `students_not_cheating`. If your model uses different names, override them:

```bash
LABEL_CHEATING=cheating LABEL_OK=not_cheating python -m src.gui --model ./models/custom.pt
```

---

### Database locked error

SQLite supports only a single writer at a time. Ensure no other process (e.g., a SQLite browser tool) has the database open simultaneously.

---

For more detailed troubleshooting and usage examples see [`docs/usage.md`](docs/usage.md).  
For architecture details, training instructions, and contribution guidelines see [`docs/development.md`](docs/development.md).

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository and create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes.** Follow the code style guidelines:
   - Adhere to [PEP 8](https://peps.python.org/pep-0008/).
   - Use type hints and docstrings for public functions.
   - Keep performance-sensitive work off the GUI thread.
   - No hardcoded absolute paths — use CLI args, environment variables, or paths relative to `__file__`.

3. **Test your changes** (run `simulate_detections.py` to verify strike logic without a camera).

4. **Open a pull request** with a clear description of what you changed and why.

**Branch naming conventions:**

| Type | Pattern | Example |
|------|---------|---------|
| New feature | `feature/<name>` | `feature/export-csv-logs` |
| Bug fix | `fix/<name>` | `fix/camera-index-crash` |
| Documentation | `docs/<name>` | `docs/update-usage-guide` |

Please open an [issue](https://github.com/Unknown4730/anticheat/issues) for bug reports or feature requests before starting large changes.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
