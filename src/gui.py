"""
Anti-Cheat Surveillance GUI
============================
Detects exam cheating in real-time using YOLOv8 + BoT-SORT tracking.

Usage:
    python src/gui.py [--model PATH] [--config PATH] [--db PATH] [--confidence FLOAT] [--max-strikes INT]

Environment variables (override defaults):
    MODEL_PATH           Path to YOLO weights file
    TRACKER_CONFIG       Path to BoT-SORT tracker YAML
    DB_PATH              Path to SQLite database file
    CONFIDENCE_THRESHOLD Detection confidence threshold (float, default: 0.5)
    MAX_STRIKES          Strikes before flagging a student (int, default: 3)
"""

import argparse
import os
import sys
import cv2
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------------------------------------------------------------
# Determine repository root (two levels up from this file: src/ -> repo root)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Default paths (relative to repo root; overridable via CLI or env vars)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = os.path.join(_REPO_ROOT, "models", "best.pt")
_DEFAULT_TRACKER = os.path.join(_REPO_ROOT, "config", "botsort.yaml")
_DEFAULT_DB = os.path.join(_REPO_ROOT, "cheat_logs.db")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Anti-Cheat Surveillance – YOLOv8 + BoT-SORT"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", _DEFAULT_MODEL),
        help="Path to YOLOv8 weights file (default: models/best.pt)",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("TRACKER_CONFIG", _DEFAULT_TRACKER),
        help="Path to BoT-SORT tracker config YAML (default: config/botsort.yaml)",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("DB_PATH", _DEFAULT_DB),
        help="Path to SQLite database file (default: cheat_logs.db at repo root)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5")),
        help="Minimum detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--max-strikes",
        type=int,
        default=int(os.environ.get("MAX_STRIKES", "3")),
        help="Number of strikes before a student is flagged (default: 3)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Runtime settings (non-path tuning parameters; overridden by parse_args())
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5
MAX_STRIKES = 3


def validate_paths(model_path, tracker_config):
    """Validate that required files exist before loading."""
    if not os.path.isfile(model_path):
        print(
            f"ERROR: Model file not found: {model_path}\n"
            "Place your trained weights at models/best.pt or pass --model <path>."
        )
        sys.exit(1)
    if not os.path.isfile(tracker_config):
        print(
            f"ERROR: Tracker config not found: {tracker_config}\n"
            "Expected at config/botsort.yaml or pass --config <path>."
        )
        sys.exit(1)


def init_db(db_path):
    """Create/connect to SQLite DB and ensure the detections table exists."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
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
    return conn, cursor


# ---------------------------------------------------------------------------
# Globals filled after argument parsing
# ---------------------------------------------------------------------------
model = None
conn = None
cursor = None
TRACKER_CONFIG = None

strike_counts = defaultdict(int)
cap = None
running = False


# ---------------------------------------------------------------------------
# Webcam utilities
# ---------------------------------------------------------------------------
def list_webcams(max_devices=5):
    available = []
    for i in range(max_devices):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            available.append(i)
            temp_cap.release()
    return available


# ---------------------------------------------------------------------------
# GUI action handlers
# ---------------------------------------------------------------------------
def start_detection():
    global cap, running
    try:
        selected_index = int(cam_select.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid camera index selected.")
        return

    cap = cv2.VideoCapture(selected_index)
    if not cap.isOpened():
        messagebox.showerror(
            "Error", f"Webcam at index {selected_index} could not be accessed."
        )
        return

    running = True
    threading.Thread(target=run_detection, daemon=True).start()


def stop_detection():
    global running
    running = False


def reset_logs():
    global strike_counts
    strike_counts.clear()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    messagebox.showinfo("Logs Reset", "All cheating logs have been cleared.")


# ---------------------------------------------------------------------------
# Detection loop (runs in background thread)
# ---------------------------------------------------------------------------
def run_detection():
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
            conf=CONFIDENCE_THRESHOLD,
        )

        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.id is None:
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
                            INSERT INTO detections
                                (student_id, timestamp, label, strikes)
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
        cv2.imshow("Anti-Cheat Detection (BoT-SORT)", frame_large)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global model, conn, cursor, TRACKER_CONFIG, CONFIDENCE_THRESHOLD, MAX_STRIKES

    args = parse_args()
    validate_paths(args.model, args.config)

    TRACKER_CONFIG = args.config
    CONFIDENCE_THRESHOLD = args.confidence
    MAX_STRIKES = args.max_strikes
    model = YOLO(args.model)
    conn, cursor = init_db(args.db)

    # Build GUI
    root = tk.Tk()
    root.title("Anti-Cheat Surveillance")
    root.geometry("400x220")
    root.configure(bg="#f0f0f0")

    tk.Label(root, text="Select Webcam:", font=("Arial", 12), bg="#f0f0f0").pack(
        pady=10
    )

    global cam_select
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

    stop_detection()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    main()
