# Usage Guide

## Running the GUI

```bash
python src/gui.py
```

### Options

| Flag / Variable | Default | Description |
|---|---|---|
| `--model <path>` | `models/yolov8n.pt` | Path to YOLO weights file |
| `MODEL_PATH` env var | `models/yolov8n.pt` | Alternative to `--model` |

The environment variable takes precedence over the CLI flag; both override the default.

---

## GUI Controls

Once the application window opens:

1. **Select Webcam** – choose the camera index from the dropdown (0 is typically the built-in webcam).
2. **Start Detection** – opens a live OpenCV window and begins tracking.
3. **Reset Logs** – clears all strike counters and removes all rows from the SQLite database.
4. **Quit** – stops detection and closes the application.

While the detection window is open, press `q` to stop the video feed.

---

## Running the Headless Script

`src/final3.py` provides the same detection logic without a GUI:

```bash
python src/final3.py
```

Keyboard shortcuts:

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset all logs |

---

## Detection Logic

- Each tracked person is assigned a persistent ID by BoT-SORT.
- Every frame where a person is classified as `students_cheating` increments their strike counter.
- When the counter reaches **MAX_STRIKES** (default: 3), a red "WARNING: Cheating Detected" overlay is shown.
- All cheating events (up to MAX_STRIKES per student) are logged to `data/cheat_logs.db`.

---

## Database Schema

The SQLite database (`data/cheat_logs.db`) is created automatically:

```sql
CREATE TABLE detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    timestamp  TEXT,
    label      TEXT,
    strikes    INTEGER
);
```

You can inspect it with any SQLite browser or:

```bash
sqlite3 data/cheat_logs.db "SELECT * FROM detections;"
```

---

## Configuration Files

### `config/botsort.yaml`

Controls the BoT-SORT tracker behaviour (thresholds, buffer size, motion compensation).  
See the [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/) for field descriptions.

### `config/data.yaml`

Defines the dataset paths and class names used during training.  
Paths are relative to the `config/` directory.

---

## Large Model Files and Git LFS

`models/yolov8n.pt` (~6 MB) is small enough to commit directly.  
If you add a larger fine-tuned model (> 50 MB), consider using Git Large File Storage:

```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git add models/your_large_model.pt
git commit -m "Add fine-tuned weights via LFS"
```
