# Anticheat — Classroom Monitoring (Refactored)

A lightweight classroom monitoring tool that uses a YOLO-based detector + tracker to identify students and flag suspected cheating. The system supports multi-student detection, per-student strike counting (3-strike rule by default), a simple GUI, and an audit log of detection events.

Table of contents
- Overview
- Features
- Prerequisites
- Installation
- Quickstart
- Configuration
- Running the GUI
- Dataset & model layout
- Database & logs
- Troubleshooting
- Contributing
- License

Overview
--------
This project detects students in a classroom-like video feed and classifies each detection as either "cheating" or "not cheating" (based on your model's output). Each student is assigned an ID (from the tracker or a short-term centroid-based fallback), and strikes are counted per student. When a student reaches the configured strike limit (default 3), they are flagged as cheating.

Features
--------
- Multi-student detection and tracking for classroom environments.
- Per-student strike counters with configurable MAX_STRIKES (default 3).
- Uses ultralytics YOLO model for detection and can use tracker IDs when available.
- GUI with side panel showing active students, strike counts and statuses.
- Exports flagged students to CSV and persists detection events in a runtime SQLite DB.
- Configurable thresholds and model path via CLI args or environment variables.

Prerequisites
-------------
- Python 3.8 or newer
- Recommended: GPU + compatible CUDA and matching PyTorch build for realtime performance
- System requirements for OpenCV and ultralytics (see ultralytics docs)

Installation
------------
1. Clone the repository:
   ```
   git clone https://github.com/Unknown4730/anticheat.git
   cd anticheat
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```

Quickstart
----------
1. Ensure a model exists at `./models/yolov8n.pt` or provide a different model path.
2. Run the GUI (default camera index 0):
   ```
   python -m src.gui --model ./models/yolov8n.pt --conf 0.5 --max-strikes 3
   ```
3. The video feed is shown in an OpenCV window. Use the GUI controls to Start/Stop and Reset logs. Press `q` in the video window to stop the detector.

Configuration
-------------
You can configure the detector via command-line arguments or environment variables.

CLI arguments (see `python -m src.gui --help`):
- `--model`, `-m` : Path to model weights (default: `./models/yolov8n.pt` or `MODEL_PATH` env var)
- `--conf` : Confidence threshold for detections (default: `0.5`)
- `--max-strikes` : Maximum strikes before a student is flagged (default: `3`)
- `--tracker-config` : Tracker configuration file path (default: `config/botsort.yaml`)

Environment variables:
- MODEL_PATH — default model path used if `--model` is not set
- CONFIDENCE_THRESHOLD — default confidence threshold
- MAX_STRIKES — default strike limit
- MOVEMENT_THRESHOLD_PIXELS — centroid movement threshold (pixels) for movement detection
- LABEL_CHEATING — model label for cheating detections (default: `students_cheating`)
- LABEL_OK — model label for non-cheating detections (default: `students_not_cheating`)

Running the GUI
---------------
- The GUI (`src/gui.py`) is the primary entrypoint. It will:
  - Create a runtime SQLite DB at `runtime/cheat_logs.db` (if not present).
  - Start camera capture and run detection/tracking in a background thread.
  - Show detection results and per-student status in a side panel.
  - Allow exporting flagged students (status = cheating) to CSV.
- For headless testing or CI, use `scripts/simulate_detections.py` to validate strike logic without hardware.

Dataset & model layout
----------------------
Recommended repository structure after refactor:
```
.
├── src/                # main application code (gui.py, helpers)
├── models/             # model weights (e.g. yolov8n.pt)
├── data/
│   ├── train/
│   └── valid/
├── config/             # botsort.yaml and other configs
├── runtime/            # runtime DB & logs (ignored by git)
├── docs/               # documentation (usage/development)
├── requirements.txt
└── README.md
```

- If your model weights are large, consider using Git LFS:
  - `git lfs install`
  - `git lfs track "models/*.pt"`
  - Commit the `.gitattributes` file.

Database & logs
---------------
- Detection events are persisted to `runtime/cheat_logs.db` in the `detections` table:
  - Columns: id, student_id, timestamp, label, strikes
- GUI "Reset Logs" clears the `detections` table and resets in-memory counters.
- Runtime files and DBs are ignored by `.gitignore` to avoid committing ephemeral data.

Troubleshooting
---------------
- Model load error: Confirm the model path provided exists and that ultralytics + torch versions are compatible.
- ultralytics import error: Install/update ultralytics and torch as per the versions in `requirements.txt`.
- Camera not found: Change the camera index in the GUI or verify camera permissions.
- Wrong labels/behavior: Ensure your model outputs the configured labels (LABEL_CHEATING / LABEL_OK). If different, set these environment variables or adapt the model.

Links
-----
- More usage details: `docs/usage.md`
- Development notes: `docs/development.md`

Contributing
------------
- Please open issues for problems or feature requests.
- Follow the project style:
  - Type hints and docstrings encouraged.
  - Keep performance-sensitive work off the GUI thread.
  - Add tests under `tests/` for logic (e.g., strike accumulation rules).
- To contribute:
  1. Fork the repo, create a feature branch.
  2. Make changes and add/update tests.
  3. Open a pull request with a clear description.

License
-------
This repository does not include a license file by default. If you want to make the code permissively reusable, add an MIT license by creating a `LICENSE` file with the standard MIT text.

If you prefer, I can: create the README in the repository, add LICENSE (MIT), or open a PR with the README plus the other documentation files (usage, development).
