"""
Anti-Cheat Surveillance GUI
============================
Detects exam cheating in real-time using YOLOv8 + BoT-SORT tracking.

Supports multiple simultaneous students (classroom environment), per-student
3-strike rule, colour-coded bounding boxes, a live status panel, and CSV export.

Usage:
    python src/gui.py [--model PATH] [--config PATH] [--db PATH]
                      [--conf-thresh FLOAT] [--max-strikes INT]

    python -m src.gui --model ./models/best.pt --max-strikes 3

Environment variables (override defaults):
    MODEL_PATH             Path to YOLO weights file
    TRACKER_CONFIG         Path to BoT-SORT tracker YAML
    DB_PATH                Path to SQLite database file
    CONFIDENCE_THRESHOLD   Minimum detection confidence (0–1)
    MAX_STRIKES            Strikes before a student is flagged as cheating
    MOVEMENT_THRESHOLD_PX  Pixel displacement counted as movement

Label mapping (override via env vars if your model uses different names):
    CONFIG_LABEL_CHEATING      default: "students_cheating"
    CONFIG_LABEL_NOT_CHEATING  default: "students_not_cheating"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import sqlite3
import threading
from typing import Optional, Tuple

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Pure detection logic (importable without GUI/CV dependencies)
from src.detector import (
    StudentState,
    STATUS_OK,
    STATUS_WARNING,
    STATUS_CHEATING,
    CONFIG_LABEL_CHEATING,
    CONFIG_LABEL_NOT_CHEATING,
    _nearest_centroid_id,
    init_db,
)

# ---------------------------------------------------------------------------
# Configurable constants – override via environment variables or CLI flags
# ---------------------------------------------------------------------------

#: Minimum YOLO detection confidence required before processing a box.
CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

#: Number of cheating strike events before a student is marked as an
#: absolute cheater.
MAX_STRIKES: int = int(os.environ.get("MAX_STRIKES", "3"))

#: Pixel displacement of a student's centroid between frames that is
#: counted as a movement event.
MOVEMENT_THRESHOLD_PIXELS: int = int(os.environ.get("MOVEMENT_THRESHOLD_PX", "30"))

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
_RUNTIME_DIR = os.path.join(_REPO_ROOT, "runtime")
_DEFAULT_DB = os.path.join(_RUNTIME_DIR, "cheat_logs.db")

# Bounding-box colours (BGR for OpenCV)
_COLOR_OK = (0, 200, 0)         # green
_COLOR_WARNING = (0, 165, 255)  # orange
_COLOR_CHEATING = (0, 0, 255)   # red


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, falling back to environment variables."""
    parser = argparse.ArgumentParser(
        description="Anti-Cheat Surveillance – YOLOv8 + BoT-SORT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", _DEFAULT_MODEL),
        help="Path to YOLOv8 weights file.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("TRACKER_CONFIG", _DEFAULT_TRACKER),
        help="Path to BoT-SORT tracker config YAML.",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("DB_PATH", _DEFAULT_DB),
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        dest="conf_thresh",
        help="Minimum detection confidence (0–1).",
    )
    parser.add_argument(
        "--max-strikes",
        type=int,
        default=MAX_STRIKES,
        dest="max_strikes",
        help="Number of cheating strikes before flagging a student.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path & startup validation
# ---------------------------------------------------------------------------

def validate_paths(model_path: str, tracker_config: str) -> None:
    """Validate that required files exist before loading the model.

    Args:
        model_path:     Path to the YOLOv8 weights ``.pt`` file.
        tracker_config: Path to the BoT-SORT YAML configuration.

    Exits the process with a descriptive message if either file is missing.
    """
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


# ---------------------------------------------------------------------------
# Box colour helper
# ---------------------------------------------------------------------------

def _status_color(status: str) -> Tuple[int, int, int]:
    """Return the BGR colour tuple for a given status string."""
    return {
        STATUS_OK: _COLOR_OK,
        STATUS_WARNING: _COLOR_WARNING,
        STATUS_CHEATING: _COLOR_CHEATING,
    }.get(status, _COLOR_OK)


# ---------------------------------------------------------------------------
# Webcam utilities
# ---------------------------------------------------------------------------

def list_webcams(max_devices: int = 5) -> list:
    """Probe camera indices 0…max_devices-1 and return those that open."""
    available = []
    for i in range(max_devices):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            available.append(i)
            temp_cap.release()
    return available


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class AntiCheatApp:
    """Tkinter-based GUI application for the Anti-Cheat Surveillance system.

    Attributes:
        conf_thresh:    Detection confidence threshold (0–1).
        max_strikes:    Strike count that flags a student as a cheater.
        tracker_config: Path to the BoT-SORT YAML config.
        db_conn:        SQLite connection.
        db_cursor:      SQLite cursor.
        student_state:  Per-student counters and status.
        model:          Loaded YOLO model.
        cap:            OpenCV video capture.
        running:        Whether the detection thread is active.
        _gui_lock:      Mutex protecting GUI updates from the detection thread.
    """

    def __init__(
        self,
        root: tk.Tk,
        yolo_model,
        db_conn: sqlite3.Connection,
        db_cursor: sqlite3.Cursor,
        tracker_config: str,
        conf_thresh: float,
        max_strikes: int,
    ) -> None:
        self.root = root
        self.model = yolo_model
        self.db_conn = db_conn
        self.db_cursor = db_cursor
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.max_strikes = max_strikes

        self.student_state = StudentState()
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self._gui_lock = threading.Lock()
        self._next_temp_id = 1  # counter for centroid-fallback IDs

        self._build_ui()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the Tkinter window layout."""
        self.root.title("Anti-Cheat Surveillance")
        self.root.geometry("820x560")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)

        # ── Left control panel ──────────────────────────────────────────
        left = tk.Frame(self.root, bg="#f0f0f0", width=240)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)
        left.pack_propagate(False)

        tk.Label(
            left, text="Anti-Cheat Surveillance", font=("Arial", 13, "bold"),
            bg="#f0f0f0",
        ).pack(pady=(0, 10))

        # Camera selection
        tk.Label(left, text="Select Webcam:", font=("Arial", 11), bg="#f0f0f0").pack()
        available_cams = list_webcams()
        self.cam_select = ttk.Combobox(
            left,
            values=[str(i) for i in available_cams],
            state="readonly",
            width=8,
            font=("Arial", 11),
        )
        self.cam_select.set(str(available_cams[0]) if available_cams else "0")
        self.cam_select.pack(pady=(2, 10))

        # MAX_STRIKES setting
        strikes_frame = tk.Frame(left, bg="#f0f0f0")
        strikes_frame.pack(pady=(0, 10))
        tk.Label(strikes_frame, text="Max Strikes:", font=("Arial", 11),
                 bg="#f0f0f0").pack(side=tk.LEFT)
        self.strikes_var = tk.StringVar(value=str(self.max_strikes))
        tk.Spinbox(
            strikes_frame, from_=1, to=10, textvariable=self.strikes_var,
            width=4, font=("Arial", 11),
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Buttons
        btn_cfg = {"font": ("Arial", 11), "width": 20, "pady": 4}
        tk.Button(
            left, text="▶  Start Detection", command=self.start_detection,
            bg="#4CAF50", fg="white", **btn_cfg,
        ).pack(pady=(0, 6))
        tk.Button(
            left, text="■  Stop Detection", command=self.stop_detection,
            bg="#2196F3", fg="white", **btn_cfg,
        ).pack(pady=(0, 6))
        tk.Button(
            left, text="↺  Reset Session", command=self.reset_logs,
            bg="#FF9800", fg="white", **btn_cfg,
        ).pack(pady=(0, 6))
        tk.Button(
            left, text="⬇  Export CSV", command=self.export_csv,
            bg="#9C27B0", fg="white", **btn_cfg,
        ).pack(pady=(0, 6))
        tk.Button(
            left, text="✕  Quit", command=self._quit,
            bg="#F44336", fg="white", **btn_cfg,
        ).pack(pady=(10, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(
            left, textvariable=self.status_var, font=("Arial", 10, "italic"),
            bg="#f0f0f0", fg="#555",
        ).pack(pady=(12, 0))

        # ── Right panel: student table ───────────────────────────────────
        right = tk.Frame(self.root, bg="#ffffff", relief=tk.SUNKEN, bd=1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        tk.Label(
            right, text="Live Student Status", font=("Arial", 12, "bold"),
            bg="#ffffff",
        ).pack(pady=(8, 4))

        # Treeview table
        cols = ("ID", "Strikes", "Movement", "Status")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)

        # Colour tags
        self.tree.tag_configure(STATUS_OK, background="#d4edda")
        self.tree.tag_configure(STATUS_WARNING, background="#fff3cd")
        self.tree.tag_configure(STATUS_CHEATING, background="#f8d7da")

        scroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8)

        # Flagged-student label
        self.flagged_var = tk.StringVar(value="Flagged: none")
        tk.Label(
            right, textvariable=self.flagged_var, font=("Arial", 10),
            bg="#ffffff", fg="#c00",
        ).pack(pady=(0, 8))

    # ------------------------------------------------------------------
    # Control actions
    # ------------------------------------------------------------------

    def start_detection(self) -> None:
        """Open the selected webcam and start the detection background thread."""
        if self.running:
            messagebox.showinfo("Info", "Detection is already running.")
            return
        try:
            selected_index = int(self.cam_select.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid camera index selected.")
            return

        # Apply current max_strikes value from GUI spinbox
        try:
            self.max_strikes = int(self.strikes_var.get())
        except ValueError:
            pass

        self.cap = cv2.VideoCapture(selected_index)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Error",
                f"Webcam at index {selected_index} could not be accessed.",
            )
            return

        self.running = True
        self.status_var.set("Detection running…")
        threading.Thread(target=self._run_detection, daemon=True).start()

    def stop_detection(self) -> None:
        """Signal the detection thread to stop."""
        self.running = False
        self.status_var.set("Stopped.")

    def reset_logs(self) -> None:
        """Clear in-memory state and delete all DB rows."""
        self.stop_detection()
        self.student_state.clear()
        self._next_temp_id = 1
        self.db_cursor.execute("DELETE FROM detections")
        self.db_conn.commit()
        self._refresh_table()
        self.flagged_var.set("Flagged: none")
        self.status_var.set("Session reset.")
        messagebox.showinfo("Session Reset", "All cheating logs have been cleared.")

    def export_csv(self) -> None:
        """Export flagged-student data to a CSV file chosen by the user."""
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export flagged students",
        )
        if not path:
            return
        flagged = [
            sid
            for sid, st in self.student_state.status.items()
            if st == STATUS_CHEATING
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["student_id", "strikes", "movement_events", "status"])
                for sid in flagged:
                    writer.writerow([
                        sid,
                        self.student_state.strike_counts[sid],
                        self.student_state.movement_counts[sid],
                        STATUS_CHEATING,
                    ])
            messagebox.showinfo(
                "Export Complete",
                f"Exported {len(flagged)} flagged student(s) to:\n{path}",
            )
        except OSError as exc:
            messagebox.showerror("Export Error", str(exc))

    def _quit(self) -> None:
        """Stop detection and close the application."""
        self.stop_detection()
        self.root.destroy()

    # ------------------------------------------------------------------
    # GUI table refresh (called from detection thread via root.after)
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        """Rebuild the student-status Treeview from current state.

        Must be called on the main Tkinter thread (use ``root.after``).
        """
        for row in self.tree.get_children():
            self.tree.delete(row)

        with self._gui_lock:
            all_ids = set(self.student_state.strike_counts.keys()) | set(
                self.student_state.status.keys()
            )
            flagged_ids = []
            for sid in sorted(all_ids):
                strikes = self.student_state.strike_counts[sid]
                movement = self.student_state.movement_counts[sid]
                st = self.student_state.status.get(
                    sid,
                    self.student_state.compute_status(sid, self.max_strikes),
                )
                label = f"ID {sid}" if sid > 0 else f"Tmp {-sid}"
                self.tree.insert(
                    "", tk.END,
                    values=(label, strikes, movement, st.upper()),
                    tags=(st,),
                )
                if st == STATUS_CHEATING:
                    flagged_ids.append(label)

        if flagged_ids:
            self.flagged_var.set("Flagged: " + ", ".join(flagged_ids))
        else:
            self.flagged_var.set("Flagged: none")

    # ------------------------------------------------------------------
    # Detection loop (background thread)
    # ------------------------------------------------------------------

    def _run_detection(self) -> None:
        """Core detection loop.  Runs in a daemon thread.

        For each frame:
        1. Run ``model.track()`` to get bounding boxes with class/conf/ID.
        2. Skip boxes below ``conf_thresh``.
        3. Assign per-student ID (tracker ID or centroid fallback).
        4. Update movement counter.
        5. Increment strike counter if label == CONFIG_LABEL_CHEATING.
        6. Draw colour-coded bounding box and status overlay.
        7. Schedule a GUI table refresh on the main thread.
        """
        db_path = self.db_conn.execute("PRAGMA database_list").fetchone()[2]
        if db_path:
            # Regular file-based database: log alongside the DB file.
            log_file = os.path.join(os.path.dirname(os.path.abspath(db_path)), "alerts.log")
        else:
            # In-memory database (e.g., tests): use the runtime directory.
            log_file = os.path.join(_RUNTIME_DIR, "alerts.log")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.track(
                source=frame,
                stream=True,
                tracker=self.tracker_config,
                persist=True,
                conf=self.conf_thresh,
            )

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                use_tracker_ids = boxes.id is not None

                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf < self.conf_thresh:
                        continue  # Skip low-confidence detections

                    cls = int(boxes.cls[i])
                    label = self.model.names[cls]
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # ── Assign student ID ──────────────────────────────
                    if use_tracker_ids:
                        student_id = int(boxes.id[i])
                    else:
                        # Centroid-based fallback matching
                        with self._gui_lock:
                            student_id, self._next_temp_id = _nearest_centroid_id(
                                cx, cy,
                                self.student_state.last_positions,
                                self._next_temp_id,
                            )

                    # ── Update movement ────────────────────────────────
                    with self._gui_lock:
                        self.student_state.update_movement(
                            student_id, cx, cy, MOVEMENT_THRESHOLD_PIXELS
                        )

                    # ── Update strikes ─────────────────────────────────
                    with self._gui_lock:
                        if label == CONFIG_LABEL_CHEATING:
                            new_status = self.student_state.record_strike(
                                student_id,
                                self.max_strikes,
                                self.db_cursor,
                                self.db_conn,
                                label,
                                log_file=log_file,
                            )
                        else:
                            # Non-cheating frame: refresh status without
                            # incrementing (no strike decay by default).
                            new_status = self.student_state.compute_status(
                                student_id, self.max_strikes
                            )
                            self.student_state.status[student_id] = new_status

                    # ── Draw bounding box ─────────────────────────────
                    color = _status_color(new_status)
                    strikes = self.student_state.strike_counts[student_id]
                    id_label = (
                        str(student_id) if student_id > 0 else f"T{-student_id}"
                    )
                    overlay_text = (
                        f"ID {id_label} | strikes: {strikes} | {new_status.upper()}"
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, overlay_text,
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )

                    # Extra on-frame alert for confirmed cheaters
                    if new_status == STATUS_CHEATING:
                        cv2.putText(
                            frame, "!! CHEATING DETECTED",
                            (x1, y2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_CHEATING, 2,
                        )

            # Schedule GUI table refresh on the main thread
            self.root.after(0, self._refresh_table)

            cv2.imshow("Anti-Cheat Detection (BoT-SORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
        self.root.after(0, lambda: self.status_var.set("Detection stopped."))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, initialise model and DB, then launch the GUI."""
    args = parse_args()

    validate_paths(args.model, args.config)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.  Run: pip install ultralytics")
        sys.exit(1)

    yolo_model = YOLO(args.model)
    db_conn, db_cursor = init_db(args.db)

    root = tk.Tk()
    AntiCheatApp(
        root=root,
        yolo_model=yolo_model,
        db_conn=db_conn,
        db_cursor=db_cursor,
        tracker_config=args.config,
        conf_thresh=CONFIDENCE_THRESHOLD,
        max_strikes=MAX_STRIKES,
    )
    root.mainloop()

    db_conn.close()


if __name__ == "__main__":
    main()
