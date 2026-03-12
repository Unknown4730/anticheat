# Anti-Cheat Surveillance System

Real-time exam cheating detection using **YOLOv8** object detection and **BoT-SORT** multi-object tracking.  The system monitors a webcam feed, identifies students exhibiting cheating behaviour, assigns per-student **strike counts**, and logs all events to a local SQLite database.

---

## Features

- **Multi-student classroom support** – tracks up to N students simultaneously using BoT-SORT tracker IDs; falls back to centroid-based matching when tracker IDs are unavailable
- **3-strike rule** – each student gets configurable chances (`MAX_STRIKES`, default: 3) before being permanently flagged as a cheater
- **Colour-coded bounding boxes**
  - 🟢 Green → normal (`students_not_cheating`, 0 strikes)
  - 🟠 Orange → warning (1 – MAX_STRIKES-1 strikes)
  - 🔴 Red → cheating (≥ MAX_STRIKES strikes)
- **Live status panel** – side table in the GUI shows every tracked student's ID, strike count, movement count, and status in real time
- **Export CSV** – one-click export of all flagged students to a CSV file
- **Configurable thresholds** – confidence, max strikes, and movement threshold adjustable via CLI flags, environment variables, or the GUI spinbox
- **SQLite logging** – every strike event written to `runtime/cheat_logs.db`
- **Movement detection** – centroid displacement across frames counted as movement (does not add strikes by itself, but is visible in the status panel)
- **Alert log** – a plain-text `alerts.log` is written to the `runtime/` directory when a student is first confirmed as cheating

---

## Project Structure

```
anticheat/
├── src/
│   ├── gui.py              # Main application entry point (GUI + detection loop)
│   └── detector.py         # Pure-Python detection logic (importable without GUI/CV)
├── scripts/
│   └── simulate_detections.py  # Test utility – replays fake frames, 15 unit tests
├── models/
│   ├── best.pt             # Custom-trained YOLOv8 weights (Git LFS)
│   └── yolov8n.pt          # Base YOLOv8-nano pretrained weights (Git LFS)
├── data/
│   ├── train/              # Training images and labels
│   ├── valid/              # Validation images and labels
│   ├── test/               # Test images and labels
│   └── data.yaml           # Dataset configuration for YOLOv8
├── config/
│   └── botsort.yaml        # BoT-SORT tracker configuration
├── docs/
│   ├── usage.md            # Detailed usage guide
│   └── development.md      # Training, architecture, and contribution guide
├── runtime/                # Created at runtime (gitignored)
│   ├── cheat_logs.db       # SQLite detection log
│   └── alerts.log          # Plain-text cheating alert log
├── requirements.txt        # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9+
- A working webcam
- (Recommended) NVIDIA GPU with CUDA for real-time performance

### 2. Clone the repository

```bash
git lfs install
git clone https://github.com/Unknown4730/anticheat.git
cd anticheat
```

> **Git LFS required** – the model `.pt` files are stored with Git LFS.

### 3. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (CUDA) users:** Install the CUDA-enabled PyTorch build first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Then run `pip install -r requirements.txt`.

### 5. Run the application

```bash
python src/gui.py
```

The GUI will open.  Select your webcam, adjust **Max Strikes** if desired, then click **▶ Start Detection**.

---

## Configuration

All settings are configurable via **CLI flags**, **environment variables**, or the **GUI spinbox**.

| Option | CLI flag | Env var | Default |
|--------|----------|---------|---------|
| Model weights | `--model` | `MODEL_PATH` | `models/best.pt` |
| Tracker config | `--config` | `TRACKER_CONFIG` | `config/botsort.yaml` |
| Database path | `--db` | `DB_PATH` | `runtime/cheat_logs.db` |
| Confidence threshold | `--conf-thresh` | `CONFIDENCE_THRESHOLD` | `0.5` |
| Max strikes | `--max-strikes` | `MAX_STRIKES` | `3` |
| Movement threshold (px) | — | `MOVEMENT_THRESHOLD_PX` | `30` |
| Cheating label | — | `CONFIG_LABEL_CHEATING` | `students_cheating` |
| Not-cheating label | — | `CONFIG_LABEL_NOT_CHEATING` | `students_not_cheating` |

### Example commands

```bash
# Basic run with defaults
python src/gui.py

# Custom model path
python src/gui.py --model /path/to/custom_weights.pt

# Stricter confidence + 5 strikes allowed
python src/gui.py --conf-thresh 0.7 --max-strikes 5

# Override model and strike count via environment variables
MODEL_PATH=./models/best.pt MAX_STRIKES=3 python src/gui.py

# Full example from the module
python -m src.gui --model ./models/best.pt --max-strikes 3
```

---

## Running the Detection Simulator / Tests

A simulation script is included that exercises the strike logic without a webcam or YOLO model:

```bash
# Narrative walkthrough + all 15 unit tests
python scripts/simulate_detections.py

# Run as unittest only
python -m unittest scripts/simulate_detections.py -v
```

---

## GUI Walkthrough

| Control | Description |
|---------|-------------|
| **Select Webcam** | Drop-down of available camera devices |
| **Max Strikes** | Spinbox to set the strike threshold (1–10) |
| **▶ Start Detection** | Opens the webcam and starts the detection loop |
| **■ Stop Detection** | Pauses detection without resetting counters |
| **↺ Reset Session** | Clears all counters and deletes DB rows |
| **⬇ Export CSV** | Saves flagged students to a CSV file |
| **✕ Quit** | Stops detection and exits |

The **Live Student Status** panel on the right updates in real time, showing each detected student's ID, strike count, movement count, and status (colour-coded green/orange/red).

Press **q** inside the video window to stop detection.

---

## Documentation

- **[Usage Guide](docs/usage.md)** – detailed walkthrough, settings, and troubleshooting
- **[Development Guide](docs/development.md)** – training the model, architecture, dataset details, and contribution guidelines

---

## Dataset

The training dataset originates from [Roboflow Universe](https://universe.roboflow.com/rtjhx/exam-cheating-jnmv1-hodrg/dataset/1) and contains three classes:

| Class | Description |
|-------|-------------|
| `person` | Generic person detection |
| `students_cheating` | Student exhibiting cheating behaviour |
| `students_not_cheating` | Student behaving normally |

---

## License

This project is licensed under the [MIT License](LICENSE).
