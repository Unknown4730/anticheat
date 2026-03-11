# Anti-Cheat Surveillance System

Real-time exam cheating detection using **YOLOv8** object detection and **BoT-SORT** multi-object tracking. The system monitors a webcam feed, identifies students exhibiting cheating behaviour, assigns per-student strike counts, and logs all events to a local SQLite database.

---

## Features

- Real-time detection via webcam using a custom-trained YOLOv8 model
- Multi-object tracking with BoT-SORT (stable student IDs across frames)
- Three-strike system: students are flagged after repeated cheating detections
- SQLite logging of every detection event
- Simple Tkinter GUI: select camera, start/stop detection, reset logs

---

## Project Structure

```
anticheat/
├── src/
│   └── gui.py              # Main application entry point
├── models/
│   ├── best.pt             # Custom-trained YOLOv8 weights
│   └── yolov8n.pt          # Base YOLOv8-nano pretrained weights
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
git clone https://github.com/Unknown4730/anticheat.git
cd anticheat
```

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

The GUI will open. Select your webcam from the dropdown, then click **Start Detection**.

---

## Configuration

All paths are configurable via **command-line arguments** or **environment variables**.

| Option | CLI flag | Env var | Default |
|--------|----------|---------|---------|
| Model weights | `--model` | `MODEL_PATH` | `models/best.pt` |
| Tracker config | `--config` | `TRACKER_CONFIG` | `config/botsort.yaml` |
| Database path | `--db` | `DB_PATH` | `cheat_logs.db` |

### Examples

```bash
# Use a different model
python src/gui.py --model /path/to/custom_weights.pt

# Override model via environment variable
MODEL_PATH=/path/to/weights.pt python src/gui.py

# Custom database location
python src/gui.py --db /var/logs/anticheat.db
```

---

## Documentation

- **[Usage Guide](docs/usage.md)** – Detailed walkthrough of the GUI, settings, and troubleshooting
- **[Development Guide](docs/development.md)** – Training the model, architecture overview, dataset details, and contribution guidelines

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
