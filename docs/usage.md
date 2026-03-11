# Usage Guide

This guide covers everything you need to run the Anti-Cheat Surveillance system.

---

## Running the Application

From the repository root:

```bash
python src/gui.py
```

### Command-line Options

```
python src/gui.py [--model PATH] [--config PATH] [--db PATH] [--confidence FLOAT] [--max-strikes INT]

  --model PATH          Path to YOLOv8 weights (.pt). Default: models/best.pt
  --config PATH         Path to BoT-SORT tracker YAML. Default: config/botsort.yaml
  --db PATH             Path to SQLite database. Default: cheat_logs.db
  --confidence FLOAT    Minimum detection confidence (0.0–1.0). Default: 0.5
  --max-strikes INT     Strikes before flagging a student as cheating. Default: 3
```

### Environment Variables

You can also set defaults via environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `MODEL_PATH` | YOLOv8 weights file | `models/best.pt` |
| `TRACKER_CONFIG` | BoT-SORT YAML | `config/botsort.yaml` |
| `DB_PATH` | SQLite database path | `cheat_logs.db` |
| `CONFIDENCE_THRESHOLD` | Detection confidence threshold | `0.5` |
| `MAX_STRIKES` | Strikes before flagging | `3` |

---

## GUI Walkthrough

When the application starts you will see a simple control panel:

1. **Select Webcam** – Drop-down listing all available camera devices (0, 1, 2, …). Select the one connected to your exam room.
2. **Start Detection** – Opens the selected webcam and begins the detection loop. A second window titled *Anti-Cheat Detection (BoT-SORT)* will appear showing the annotated live feed.
3. **Reset Logs** – Clears the in-memory strike counter and deletes all rows from the database. Useful when starting a new exam session.
4. **Quit** – Stops detection and closes all windows.

---

## Live Feed Window

The feed window overlays bounding boxes on each detected person:

| Colour | Meaning |
|--------|---------|
| Green box | `students_not_cheating` |
| Red box | `students_cheating` |

Each box includes the class label and the tracker-assigned student ID.

When a student accumulates **3 or more strikes**, the text `WARNING: Cheating Detected` appears below their bounding box.

Press **q** inside the feed window to stop detection (same effect as the Quit button).

---

## Database & Logs

Detections are written to a SQLite database (default: `cheat_logs.db` in the repository root).

### Schema

```sql
CREATE TABLE detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,        -- BoT-SORT tracker ID
    timestamp  TEXT,        -- YYYY-MM-DD HH:MM:SS
    label      TEXT,        -- detected class name
    strikes    INTEGER      -- cumulative strike count for this student
);
```

### Querying the Database

```bash
sqlite3 cheat_logs.db "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 20;"
```

The database is created automatically at startup if it does not exist. It is listed in `.gitignore` and should not be committed to version control.

---

## Detection Settings

The following constants can be edited in `src/gui.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence (0–1) |
| `MAX_STRIKES` | `3` | Strikes before a student is flagged |

---

## Troubleshooting

### "Model file not found"

Ensure `models/best.pt` exists. If you are using a different weights file, pass its path via `--model`:

```bash
python src/gui.py --model path/to/your_weights.pt
```

### "Webcam could not be accessed"

- Check that no other application is using the camera.
- On Linux, verify you have permission: `ls -l /dev/video*`
- Try a different camera index in the dropdown.

### Very slow detection / low FPS

- A GPU is strongly recommended for real-time performance. Install the CUDA-enabled PyTorch build:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- On CPU, inference is slower. Lower the input resolution in your webcam settings or reduce the frame size in `run_detection()`.

### Tkinter not available (Linux)

```bash
sudo apt-get install python3-tk
```

### Class labels look wrong

The model was trained on three classes: `person`, `students_cheating`, `students_not_cheating`. If you see unexpected class names, you may be loading the wrong weights file. Use `models/best.pt` (custom trained) rather than `models/yolov8n.pt` (generic base model).

### Database locked error

SQLite allows a single writer. Make sure no other process has the database open simultaneously.
