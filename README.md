# Anticheat — Classroom Monitoring (Refactored)

This repository provides a classroom monitoring tool that detects students and flags potential cheating using a YOLO-based model and multi-object tracking.

Key features:
- Multi-student detection (handles multiple simultaneous students).
- Per-student 3-strike rule: each student has up to 3 cheating strikes before being flagged.
- GUI with side panel listing students, strike counts, and status.
- Runtime DB to persist detection events (runtime/cheat_logs.db).
- Configurable model path, confidence threshold and max strikes.

Prerequisites
- Python 3.8+
- Recommended GPU + CUDA for realtime performance (optional). See ultralytics and torch compatibility.

Quickstart
1. Create virtualenv and install deps:
   python -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

2. Place or verify model:
   - Default expected model: ./models/yolov8n.pt
   - Or pass a model explicitly:
     python -m src.gui --model ./models/your_custom.pt

3. Run the GUI:
   python -m src.gui --model ./models/yolov8n.pt --conf 0.5 --max-strikes 3

Notes
- The GUI opens an OpenCV display window for the video feed (press 'q' in that window to stop the detector).
- The runtime DB is created at runtime under `runtime/cheat_logs.db`. Runtime files are gitignored.
- If your model uses different label names than `students_cheating` and `students_not_cheating`, either rename or set LABEL_CHEATING/LABEL_OK env vars.

Project layout (after restructuring)
- src/ — main source code (gui.py)
- models/ — model weights (yolov8n.pt)
- data/ — dataset (train/ and valid/)
- config/ — botsort.yaml and other configs
- runtime/ — runtime DB and logs (ignored by git)
- docs/ — documentation and usage notes

If you'd like, I can:
- Create and open a PR with these changes.
- Adjust the GUI to embed the video preview inside Tkinter instead of using the OpenCV window.
- Add unit tests and a simulation script to validate the 3-strike logic.
