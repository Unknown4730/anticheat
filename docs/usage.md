# Usage Guide

This guide covers everything you need to run the Anti-Cheat Surveillance system.

---

## Running the Application

From the repository root:

```bash
python src/gui.py
```

Or as a module:

```bash
python -m src.gui
```

### Command-line Options

```
python src/gui.py [--model PATH] [--config PATH] [--db PATH]
                  [--conf-thresh FLOAT] [--max-strikes INT]

  --model PATH         Path to YOLOv8 weights (.pt). Default: models/best.pt
  --config PATH        Path to BoT-SORT tracker YAML. Default: config/botsort.yaml
  --db PATH            Path to SQLite database. Default: runtime/cheat_logs.db
  --conf-thresh FLOAT  Minimum detection confidence (0–1). Default: 0.5
  --max-strikes INT    Strikes before flagging a student. Default: 3
```

### Environment Variables

You can also set defaults via environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `MODEL_PATH` | YOLOv8 weights file | `models/best.pt` |
| `TRACKER_CONFIG` | BoT-SORT YAML | `config/botsort.yaml` |
| `DB_PATH` | SQLite database path | `runtime/cheat_logs.db` |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.5` |
| `MAX_STRIKES` | Strikes before flagging | `3` |
| `MOVEMENT_THRESHOLD_PX` | Pixel displacement counted as movement | `30` |
| `CONFIG_LABEL_CHEATING` | Model label for cheating | `students_cheating` |
| `CONFIG_LABEL_NOT_CHEATING` | Model label for normal behaviour | `students_not_cheating` |

### Example Commands

```bash
# Basic run
python src/gui.py

# Use a different model
python src/gui.py --model /path/to/custom_weights.pt

# Stricter confidence, more strikes allowed
python src/gui.py --conf-thresh 0.7 --max-strikes 5

# Environment variable overrides
MODEL_PATH=./models/best.pt MAX_STRIKES=3 python src/gui.py

# Full example
python -m src.gui --model ./models/best.pt --max-strikes 3
```

---

## GUI Walkthrough

When the application starts you will see a two-panel window:

### Left control panel

| Control | Description |
|---------|-------------|
| **Select Webcam** | Drop-down listing all available camera devices (0, 1, 2, …) |
| **Max Strikes** | Spinbox to set the strike threshold (1–10). Takes effect when detection starts. |
| **▶ Start Detection** | Opens the selected webcam and begins the detection loop |
| **■ Stop Detection** | Pauses detection without clearing counters |
| **↺ Reset Session** | Clears all in-memory counters and deletes all DB rows |
| **⬇ Export CSV** | Opens a save dialog and exports flagged students to CSV |
| **✕ Quit** | Stops detection and closes the application |

### Right status panel

A live table showing every detected student:

| Column | Description |
|--------|-------------|
| **ID** | Tracker-assigned ID (positive integer) or temporary centroid ID (shown as "Tmp N") |
| **Strikes** | Cumulative cheating-event count |
| **Movement** | Cumulative large-displacement movement count |
| **Status** | OK / WARNING / CHEATING (colour-coded) |

Rows are colour-coded:
- 🟢 Green background → `OK` (no strikes)
- 🟡 Yellow background → `WARNING` (1 to MAX_STRIKES-1 strikes)
- 🔴 Red background → `CHEATING` (≥ MAX_STRIKES strikes)

A "Flagged:" label below the table lists the IDs of confirmed cheating students.

---

## Live Feed Window

The detection feed window overlays bounding boxes on each detected student:

| Box colour | Meaning |
|-----------|---------|
| 🟢 Green  | `students_not_cheating`, 0 strikes |
| 🟠 Orange | 1 – (MAX_STRIKES-1) strikes (warning) |
| 🔴 Red    | ≥ MAX_STRIKES strikes (confirmed cheating) |

Each box includes: `ID <n> | strikes: <k> | <STATUS>`

When a student reaches the strike threshold, `!! CHEATING DETECTED` appears below their bounding box.

Press **q** inside the feed window to stop detection (same effect as ■ Stop).

---

## 3-Strike Rule

Each student starts with 0 strikes.

| Strikes | Status | Box Colour |
|---------|--------|-----------|
| 0 | OK | Green |
| 1 | WARNING | Orange |
| 2 | WARNING | Orange |
| ≥ 3 (default) | CHEATING | Red |

- A **strike** is recorded when the model detects `students_cheating` with confidence ≥ `CONFIDENCE_THRESHOLD` for a given student ID.
- Detections with confidence **below** the threshold are silently skipped.
- Non-cheating frames (`students_not_cheating`) do not add or remove strikes.
- **Movement** (centroid displacement > `MOVEMENT_THRESHOLD_PX` pixels) is counted separately and shown in the status panel but does **not** directly add strikes.

---

## Multi-Student Tracking

The system handles multiple simultaneous students:

1. **Tracker IDs (preferred)** – BoT-SORT assigns a stable integer ID to each tracked person. Per-student strike/movement counters are keyed by this ID.
2. **Centroid fallback** – if the tracker does not return IDs for a frame, the system uses nearest-centroid matching (within 80 px) to assign temporary IDs (`Tmp N`). This maintains continuity across frames.

---

## Exporting Flagged Students

Click **⬇ Export CSV** to save a CSV file with columns:

```
student_id, strikes, movement_events, status
```

Only students whose status is `CHEATING` are included.

---

## Database & Logs

All data is stored in the `runtime/` directory (created automatically; gitignored).

### SQLite database (`runtime/cheat_logs.db`)

```sql
CREATE TABLE detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,        -- tracker ID or temporary centroid ID
    timestamp  TEXT,        -- YYYY-MM-DD HH:MM:SS
    label      TEXT,        -- detected class name
    strikes    INTEGER      -- cumulative strike count after this event
);
```

A row is written **only when a strike is recorded** (not for every frame).

```bash
# Query recent detections
sqlite3 runtime/cheat_logs.db "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 20;"
```

### Alert log (`runtime/alerts.log`)

A plain-text log is appended when a student first crosses the `MAX_STRIKES` threshold:

```
2024-01-15 10:32:17 CHEATING_FLAGGED student_id=3 strikes=3
```

---

## Detection Settings

| Constant | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence (0–1) |
| `MAX_STRIKES` | `3` | Strikes before a student is flagged |
| `MOVEMENT_THRESHOLD_PIXELS` | `30` | Pixel displacement counted as movement |

These can be changed via CLI flags, environment variables, or (for `MAX_STRIKES`) the GUI spinbox.

---

## Running the Simulator / Unit Tests

A test utility is included that exercises the strike logic without a webcam:

```bash
# Narrative walkthrough + 15 unit tests
python scripts/simulate_detections.py

# Tests only
python -m unittest scripts/simulate_detections.py -v
```

---

## Troubleshooting

### "Model file not found"

Ensure `models/best.pt` exists (requires Git LFS).  If using a different file:

```bash
python src/gui.py --model path/to/your_weights.pt
```

### "Webcam could not be accessed"

- Check no other application is using the camera.
- On Linux: `ls -l /dev/video*` to verify permissions.
- Try a different camera index in the dropdown.

### Very slow detection / low FPS

A GPU is strongly recommended.  Install the CUDA-enabled PyTorch build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Tkinter not available (Linux)

```bash
sudo apt-get install python3-tk
```

### Class labels look wrong

The model was trained on three classes: `person`, `students_cheating`, `students_not_cheating`. If you see unexpected class names, you may be loading the wrong weights file or need to set the label environment variables:

```bash
CONFIG_LABEL_CHEATING=cheating CONFIG_LABEL_NOT_CHEATING=normal python src/gui.py
```

### Database locked error

SQLite allows a single writer.  Ensure no other process has `runtime/cheat_logs.db` open simultaneously.
