import cv2
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox

# === CONFIG ===
MODEL_PATH = "D:\\anticheat-main\\anticheat-main\\runs\\detect\\train\\weights\\best.pt"
TRACKER_CONFIG = "botsort.yaml"
MAX_STRIKES = 3

# === INIT YOLO ===
model = YOLO(MODEL_PATH)

# === DB SETUP ===
conn = sqlite3.connect("cheat_logs.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        timestamp TEXT,
        label TEXT,
        strikes INTEGER
    )
''')
conn.commit()

# === STRIKE TRACKING ===
strike_counts = defaultdict(int)
cap = None
running = False

# === DETECT CAMERAS ===
def list_webcams(max_devices=5):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

# === GUI ACTIONS ===
def start_detection():
    global cap, running
    selected_index = int(cam_select.get())

    cap = cv2.VideoCapture(selected_index)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access selected webcam.")
        return

    running = True
    threading.Thread(target=run_detection, daemon=True).start()

def stop_detection():
    global running
    running = False

def reset_logs():
    strike_counts.clear()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    messagebox.showinfo("Reset", "Logs cleared successfully.")

# === MAIN DETECTION LOOP ===
def run_detection():
    global cap, running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(source=frame, stream=True, tracker=TRACKER_CONFIG, persist=True)

        for r in results:
            boxes = r.boxes
            if boxes.id is not None:
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    label = model.names[cls]
                    conf = float(boxes.conf[i])
                    id_ = int(boxes.id[i])
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                    color = (0, 255, 0) if label == "students_not_cheating" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} | ID {id_}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if label == "students_cheating":
                        strike_counts[id_] += 1
                        strikes = strike_counts[id_]

                        if strikes <= MAX_STRIKES:
                            cursor.execute('''
                                INSERT INTO detections (student_id, timestamp, label, strikes)
                                VALUES (?, ?, ?, ?)
                            ''', (str(id_), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, strikes))
                            conn.commit()

                        if strikes >= MAX_STRIKES:
                            cv2.putText(frame, f"⚠️ Cheating Detected", (x1, y2 + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        frame_large = cv2.resize(frame, None, fx=1.5, fy=1.5)
        cv2.imshow("🛡️ Anti-Cheat Detection", frame_large)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

# === GUI SETUP ===
root = tk.Tk()
root.title("Anti-Cheat Surveillance 🛡️")
root.geometry("420x240")
root.configure(bg="#f0f0f0")

# Camera Dropdown
tk.Label(root, text="Select Webcam:", font=("Arial", 12), bg="#f0f0f0").pack(pady=10)
available_cams = list_webcams()
if not available_cams:
    messagebox.showerror("Error", "No webcams found! Plug in one and restart.")
    root.destroy()
    exit()

cam_select = ttk.Combobox(root, values=[str(i) for i in available_cams], state="readonly", width=10, font=("Arial", 12))
cam_select.set(str(available_cams[0]))
cam_select.pack()

# Buttons
tk.Button(root, text="▶ Start Detection", command=start_detection, bg="#4CAF50", fg="white", font=("Arial", 12), width=22).pack(pady=10)
tk.Button(root, text="🧹 Reset Logs", command=reset_logs, bg="#FF9800", fg="white", font=("Arial", 12), width=22).pack(pady=5)
tk.Button(root, text="❌ Quit", command=lambda: (stop_detection(), root.destroy()), bg="#F44336", fg="white", font=("Arial", 12), width=22).pack(pady=10)

root.mainloop()

# Cleanup
if cap:
    cap.release()
cv2.destroyAllWindows()
conn.close()
