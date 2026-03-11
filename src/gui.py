"""
Anti-Cheat Surveillance GUI
============================
Main entrypoint for the anti-cheat detection system.

Usage:
    python src/gui.py
    python src/gui.py --model models/yolov8n.pt
    MODEL_PATH=models/yolov8n.pt python src/gui.py

See README.md for full setup instructions.
"""

import argparse
import os
import sys
import cv2
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------------------------------------------------------------
# Resolve paths relative to the project root (one directory above src/)
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)


def _abs(rel_path: str) -> str:
    """Return an absolute path given a path relative to the project root."""
    return os.path.join(_PROJECT_ROOT, rel_path)


# ---------------------------------------------------------------------------
# Parse CLI arguments / environment variables
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anti-Cheat Surveillance – real-time exam monitoring via YOLO + BoT-SORT."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to the YOLO model weights file. "
            "Overridden by the MODEL_PATH environment variable if set. "
            "Defaults to ./models/yolov8n.pt relative to the project root."
        ),
    )
    return parser.parse_args()


args = _parse_args()

# Priority: env var > CLI flag > default
MODEL_PATH: str = (
    os.environ.get("MODEL_PATH")
    or args.model
    or _abs("models/yolov8n.pt")
)

TRACKER_CONFIG: str = _abs("config/botsort.yaml")
DB_PATH: str = _abs(os.path.join("data", "cheat_logs.db"))
CONFIDENCE_THRESHOLD: float = 0.5
MAX_STRIKES: int = 3

# ---------------------------------------------------------------------------
# Validate model file before importing ultralytics (gives a clearer error)
# ---------------------------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    print(
        f"\n[ERROR] Model file not found: {MODEL_PATH}\n"
        "  • Make sure the model weights file exists at that path.\n"
        "  • You can specify a different path with --model <path> or the MODEL_PATH env var.\n"
        "  • See README.md for download / re-training instructions.\n",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "\n[ERROR] The 'ultralytics' package is not installed.\n"
        "  Run:  pip install -r requirements.txt\n"
        "  See README.md for full setup instructions.\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = YOLO(MODEL_PATH)

# ---------------------------------------------------------------------------
# SQLite DB – stored in data/ so it is never committed (see .gitignore)
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS detections (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        timestamp  TEXT,
        label      TEXT,
        strikes    INTEGER
    )
    """
)
conn.commit()

strike_counts: defaultdict = defaultdict(int)
cap = None
running = False


# ---------------------------------------------------------------------------
# Webcam utility
# ---------------------------------------------------------------------------
def list_webcams(max_devices: int = 5) -> list:
    available = []
    for i in range(max_devices):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            available.append(i)
            temp_cap.release()
    return available


# ---------------------------------------------------------------------------
# GUI action functions
# ---------------------------------------------------------------------------
def start_detection() -> None:
    global cap, running
    selected_index = int(cam_select.get())
    cap = cv2.VideoCapture(selected_index)

    if not cap.isOpened():
        messagebox.showerror("Error", "Selected webcam could not be accessed.")
        return

    running = True
    threading.Thread(target=run_detection, daemon=True).start()


def stop_detection() -> None:
    global running
    running = False


def reset_logs() -> None:
    global strike_counts
    strike_counts.clear()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    messagebox.showinfo("Logs Reset", "All cheating logs have been cleared.")


# ---------------------------------------------------------------------------
# Detection loop (runs in a background thread)
# ---------------------------------------------------------------------------
def run_detection() -> None:
    global cap, running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            stream=True,
            tracker=TRACKER_CONFIG,
            persist=True,
        )

        for r in results:
            boxes = r.boxes
            if boxes.id is None:
                continue

            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                label = model.names[cls]
                id_ = int(boxes.id[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                color = (0, 255, 0) if label == "students_not_cheating" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} | ID {id_}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                if label == "students_cheating":
                    strike_counts[id_] += 1
                    strikes = strike_counts[id_]

                    if strikes <= MAX_STRIKES:
                        cursor.execute(
                            """
                            INSERT INTO detections (student_id, timestamp, label, strikes)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                str(id_),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                label,
                                strikes,
                            ),
                        )
                        conn.commit()

                    if strikes >= MAX_STRIKES:
                        cv2.putText(
                            frame,
                            "WARNING: Cheating Detected",
                            (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

        frame_large = cv2.resize(frame, None, fx=1.5, fy=1.5)
        cv2.imshow("Anti-Cheat Detection (BoT-SORT + ReID)", frame_large)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# GUI setup
# ---------------------------------------------------------------------------
root = tk.Tk()
root.title("Anti-Cheat Surveillance")
root.geometry("400x220")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Select Webcam:", font=("Arial", 12), bg="#f0f0f0").pack(pady=10)
available_cams = list_webcams()
cam_select = ttk.Combobox(
    root,
    values=[str(i) for i in available_cams],
    state="readonly",
    width=10,
    font=("Arial", 12),
)
cam_select.set(str(available_cams[0]) if available_cams else "0")
cam_select.pack()

tk.Button(
    root,
    text="Start Detection",
    command=start_detection,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12),
    width=20,
).pack(pady=10)
tk.Button(
    root,
    text="Reset Logs",
    command=reset_logs,
    bg="#FF9800",
    fg="white",
    font=("Arial", 12),
    width=20,
).pack(pady=5)
tk.Button(
    root,
    text="Quit",
    command=lambda: (stop_detection(), root.destroy()),
    bg="#F44336",
    fg="white",
    font=("Arial", 12),
    width=20,
).pack(pady=10)

root.mainloop()

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
if cap:
    cap.release()
cv2.destroyAllWindows()
conn.close()
