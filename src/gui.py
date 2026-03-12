#!/usr/bin/env python3
"""
Main GUI and detection entrypoint.

Key features:
- Configurable MODEL_PATH via env var MODEL_PATH or CLI --model
- CONFIDENCE_THRESHOLD and MAX_STRIKES configurable via CLI/env
- Per-ID strike tracking with 3-strike rule
- Tracker ID usage with centroid fallback matching when tracker IDs are missing
- Runtime DB created under runtime/cheat_logs.db
- Side panel showing students and strike counts, Export CSV for flagged students
"""

import os
import sys
import argparse
import sqlite3
import threading
import time
from datetime import datetime
from collections import defaultdict
import math
import csv

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ultralytics YOLO — make sure it's installed in requirements
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: ultralytics module not found. Install dependencies (see requirements.txt).")
    raise

# ---------------------------
# Defaults and configuration
# ---------------------------
DEFAULT_MODEL = os.environ.get("MODEL_PATH", "./models/yolov8n.pt")
DEFAULT_CONF = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
DEFAULT_MAX_STRIKES = int(os.environ.get("MAX_STRIKES", 3))
MOVEMENT_THRESHOLD_PIXELS = int(os.environ.get("MOVEMENT_THRESHOLD_PIXELS", 30))  # centroid movement considered notable
RUNTIME_DIR = "runtime"
DB_PATH = os.path.join(RUNTIME_DIR, "cheat_logs.db")
TRACKER_CONFIG = "config/botsort.yaml"  # ensure this file exists relative to repo root

# Labels (configurable if your model uses different names)
LABEL_CHEATING = os.environ.get("LABEL_CHEATING", "students_cheating")
LABEL_OK = os.environ.get("LABEL_OK", "students_not_cheating")

# Thread-safe structures — updated only from the detection thread
strike_counts = defaultdict(int)
last_positions = {}   # id -> (cx, cy)
status = {}           # id -> 'ok'|'warning'|'cheating'
active_ids = set()    # IDs currently seen in frames

# Internal mapping for centroid fallback when tracker doesn't provide IDs
_next_temp_id = 1000000
temp_id_lock = threading.Lock()


# ---------------------------
# Utilities
# ---------------------------
def ensure_runtime():
    os.makedirs(RUNTIME_DIR, exist_ok=True)


def init_db(path=DB_PATH):
    ensure_runtime()
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            timestamp TEXT,
            label TEXT,
            strikes INTEGER
        )
    ''')
    conn.commit()
    return conn


def record_detection(conn, student_id, label, strikes):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detections (student_id, timestamp, label, strikes) VALUES (?, ?, ?, ?)",
        (str(student_id), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, strikes),
    )
    conn.commit()


def centroid_from_xyxy(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def new_temp_id():
    global _next_temp_id
    with temp_id_lock:
        _next_temp_id += 1
        return _next_temp_id


def match_centroids(prev_positions, current_centroids, max_dist=50):
    """
    Simple greedy nearest neighbour matching.
    prev_positions: dict id -> centroid
    current_centroids: list of centroid tuples
    Returns mapping list of (assigned_id, centroid) for each centroid in current_centroids.
    Unmatched centroids get new temp ids.
    """
    assigned = []
    prev_items = list(prev_positions.items())
    used_prev = set()

    for c in current_centroids:
        best_id = None
        best_d = None
        for pid, pcent in prev_items:
            if pid in used_prev:
                continue
            d = distance(pcent, c)
            if best_d is None or d < best_d:
                best_d = d
                best_id = pid
        if best_d is not None and best_d <= max_dist:
            assigned.append((best_id, c))
            used_prev.add(best_id)
        else:
            # new ID
            assigned.append((new_temp_id(), c))
    return assigned


# ---------------------------
# Detection loop
# ---------------------------
class Detector:
    def __init__(self, model_path, conf_thresh=DEFAULT_CONF, max_strikes=DEFAULT_MAX_STRIKES, tracker_cfg=TRACKER_CONFIG):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf_thresh
        self.max_strikes = max_strikes
        self.tracker_cfg = tracker_cfg
        self.conn = init_db()
        self.running = False
        self.cap = None
        self.gui_update_callback = None  # function to update GUI list
        self.lock = threading.Lock()

    def start_camera(self, cam_index=0):
        if self.cap is not None:
            self.stop()
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_index}")
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _update_status_for_id(self, sid):
        s = strike_counts.get(sid, 0)
        if s >= self.max_strikes:
            status[sid] = "cheating"
        elif s > 0:
            status[sid] = "warning"
        else:
            status[sid] = "ok"

    def _run(self):
        global active_ids, last_positions
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            # run tracker/detector — using track to get IDs when available
            try:
                results = self.model.track(source=frame, stream=True, tracker=self.tracker_cfg, persist=True)
            except Exception as e:
                # fallback: try inference; continue loop
                results = self.model(frame, stream=True)

            # gather current detections for centroid fallback when needed
            current_centroids = []
            current_candidates = []  # store (idx, label, conf, xyxy, maybe_id)
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue

                # boxes may contain .id, .cls, .conf, .xyxy
                n = len(boxes)
                for i in range(n):
                    try:
                        conf = float(boxes.conf[i])
                    except Exception:
                        conf = 0.0
                    if conf < self.conf:
                        continue
                    try:
                        xy = boxes.xyxy[i].tolist()
                    except Exception:
                        continue
                    x1, y1, x2, y2 = map(int, xy)
                    cx, cy = centroid_from_xyxy(x1, y1, x2, y2)
                    try:
                        cls = int(boxes.cls[i])
                        label = self.model.names.get(cls, str(cls))
                    except Exception:
                        label = "unknown"
                    # ID if exist:
                    assigned_id = None
                    try:
                        if hasattr(boxes, "id") and boxes.id is not None:
                            # Some trackers expose id array, else attribute may be missing
                            assigned_id = int(boxes.id[i])
                    except Exception:
                        assigned_id = None

                    current_centroids.append((cx, cy))
                    current_candidates.append({
                        "centroid": (cx, cy),
                        "xyxy": (x1, y1, x2, y2),
                        "label": label,
                        "conf": conf,
                        "id": assigned_id,
                    })

            # If some detections have IDs, use them; others fallback to centroid matching with previous positions.
            # Build mapping of centroid->id for those without.
            # Prepare prev_positions dict from last_positions (only for recent active ids)
            prev_positions = {sid: last_positions[sid] for sid in list(last_positions.keys())}

            # For candidates without an assigned id, prepare a list to match
            centroids_to_match = []
            idxs_to_match = []
            for idx, cand in enumerate(current_candidates):
                if cand["id"] is None:
                    centroids_to_match.append(cand["centroid"])
                    idxs_to_match.append(idx)

            matched = {}
            if centroids_to_match:
                matches = match_centroids(prev_positions, centroids_to_match, max_dist=MOVEMENT_THRESHOLD_PIXELS * 2)
                # matches is list of (id, centroid) aligned with centroids_to_match
                for k, (mid, cent) in enumerate(matches):
                    idx = idxs_to_match[k]
                    matched[idx] = mid

            # Now iterate through candidates and update per-id structures
            seen_ids = set()
            for idx, cand in enumerate(current_candidates):
                sid = cand["id"] if cand["id"] is not None else matched.get(idx)
                if sid is None:
                    sid = new_temp_id()
                seen_ids.add(sid)

                cx, cy = cand["centroid"]
                label = cand["label"]
                conf = cand["conf"]
                x1, y1, x2, y2 = cand["xyxy"]

                # Initialize last_positions if new
                if sid not in last_positions:
                    last_positions[sid] = (cx, cy)

                moved = False
                prev = last_positions.get(sid)
                if prev:
                    if distance(prev, (cx, cy)) >= MOVEMENT_THRESHOLD_PIXELS:
                        moved = True

                # Update last position
                last_positions[sid] = (cx, cy)
                active_ids = seen_ids

                # Decision logic:
                # Only increment strikes on an explicit 'cheating' label with sufficient confidence.
                if label == LABEL_CHEATING:
                    # increment strike
                    strike_counts[sid] += 1
                    self._update_status_for_id(sid)
                    record_detection(self.conn, sid, label, strike_counts[sid])
                else:
                    # label == OK or other
                    # Optional: no automatic decrease. Could implement decay if desired.
                    self._update_status_for_id(sid)

                # draw boxes and overlay info on frame
                color = (0, 255, 0)  # green by default
                if status.get(sid) == "cheating":
                    color = (0, 0, 255)  # red
                elif status.get(sid) == "warning":
                    color = (0, 165, 255)  # orange

                # draw rectangle and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                txt = f"ID {sid} | strikes: {strike_counts.get(sid,0)}"
                cv2.putText(frame, txt, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Remove stale last_positions for ids not seen for a while? (left for enhancements)
            # Optionally update the GUI side panel
            if self.gui_update_callback:
                try:
                    # send snapshot of statuses
                    snapshot = []
                    for sid in list(set(list(strike_counts.keys()) + list(last_positions.keys()))):
                        snapshot.append({
                            "id": sid,
                            "strikes": strike_counts.get(sid, 0),
                            "status": status.get(sid, "ok"),
                        })
                    self.gui_update_callback(snapshot)
                except Exception:
                    pass

            # display frame in its own window (or we can embed in tkinter - simpler to show cv2 window)
            cv2.imshow("Anticheat - Press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

        # cleanup
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ---------------------------
# Simple GUI wrapper
# ---------------------------
class AnticheatApp:
    def __init__(self, detector: Detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("Anticheat Classroom Monitor")
        self._build_ui()
        self.detector.gui_update_callback = self.update_student_list

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Controls top
        ctrl = ttk.Frame(frm)
        ctrl.grid(row=0, column=0, sticky="ew", pady=4)
        ttk.Label(ctrl, text="Camera:").grid(row=0, column=0, padx=4)
        self.cam_var = tk.StringVar(value="0")
        ttk.Entry(ctrl, textvariable=self.cam_var, width=6).grid(row=0, column=1)
        ttk.Button(ctrl, text="Start", command=self.start).grid(row=0, column=2, padx=4)
        ttk.Button(ctrl, text="Stop", command=self.stop).grid(row=0, column=3, padx=4)
        ttk.Button(ctrl, text="Reset Logs", command=self.reset_logs).grid(row=0, column=4, padx=4)
        ttk.Button(ctrl, text="Export Flagged CSV", command=self.export_flagged).grid(row=0, column=5, padx=4)

        # Student list
        list_frame = ttk.Frame(frm)
        list_frame.grid(row=1, column=0, sticky="nsew")
        frm.rowconfigure(1, weight=1)
        self.tree = ttk.Treeview(list_frame, columns=("id", "strikes", "status"), show="headings", height=12)
        self.tree.heading("id", text="ID")
        self.tree.heading("strikes", text="Strikes")
        self.tree.heading("status", text="Status")
        self.tree.column("id", width=120)
        self.tree.column("strikes", width=80)
        self.tree.column("status", width=120)
        self.tree.pack(fill="both", expand=True)

    def start(self):
        try:
            cam_index = int(self.cam_var.get())
        except Exception:
            cam_index = 0
        try:
            self.detector.start_camera(cam_index)
            messagebox.showinfo("Started", "Detection started. The video shows in an OpenCV window. Press 'q' in that window to stop.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop(self):
        self.detector.stop()

    def reset_logs(self):
        global strike_counts, last_positions, status
        strike_counts.clear()
        last_positions.clear()
        status.clear()
        # clear DB table when user requests reset
        try:
            cur = self.detector.conn.cursor()
            cur.execute("DELETE FROM detections")
            self.detector.conn.commit()
            messagebox.showinfo("Reset", "All logs and counters cleared.")
            self.update_student_list([])
        except Exception as e:
            messagebox.showerror("Reset error", str(e))

    def update_student_list(self, snapshot):
        # snapshot: list of dicts with id,strikes,status
        # update treeview
        existing = {self.tree.set(k, "id"): k for k in self.tree.get_children()}
        seen_ids = set()
        for item in snapshot:
            sid = str(item["id"])
            seen_ids.add(sid)
            if sid in existing:
                self.tree.item(existing[sid], values=(sid, item["strikes"], item["status"]))
            else:
                self.tree.insert("", "end", values=(sid, item["strikes"], item["status"]))
        # remove rows not in snapshot
        for iid in list(self.tree.get_children()):
            if self.tree.set(iid, "id") not in seen_ids:
                self.tree.delete(iid)

    def export_flagged(self):
        # Export students whose status == cheating
        rows = []
        for sid, s in status.items():
            if s == "cheating":
                rows.append((sid, strike_counts.get(sid, 0)))
        if not rows:
            messagebox.showinfo("No flagged", "No flagged students at this time.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["student_id", "strikes"])
            writer.writerows(rows)
        messagebox.showinfo("Exported", f"Exported {len(rows)} flagged students to {path}")

    def run(self):
        self.root.mainloop()


# ---------------------------
# CLI Entrypoint
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Anticheat multi-student detector GUI")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Path to model weights (default from env or ./models/yolov8n.pt)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold for detections")
    p.add_argument("--max-strikes", type=int, default=DEFAULT_MAX_STRIKES, help="Max strikes before flagged as cheating")
    p.add_argument("--tracker-config", default=TRACKER_CONFIG, help="Tracker config path")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Please provide a valid path via --model or set MODEL_PATH env var.")
        sys.exit(1)
    detector = Detector(model_path=args.model, conf_thresh=args.conf, max_strikes=args.max_strikes, tracker_cfg=args.tracker_config)
    app = AnticheatApp(detector)
    app.run()

if __name__ == "__main__":
    main()
