import cv2
import sqlite3
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# --- SETTINGS ---
CONFIDENCE_THRESHOLD = 0.5
MAX_STRIKES = 3
MODEL_PATH = "runs/detect/train/weights/best.pt"
TRACKER_CONFIG = "botsort.yaml"

# --- INIT DATABASE ---
conn = sqlite3.connect("cheat_logs.db")
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

# --- INIT YOLO MODEL WITH TRACKING ---
model = YOLO(MODEL_PATH)

# --- STRIKE COUNTER ---
strike_counts = defaultdict(int)

# --- CAPTURE VIDEO ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, persist=True, conf=CONFIDENCE_THRESHOLD, tracker=TRACKER_CONFIG, stream=True)

    for r in results:
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = model.names[cls_id]
                conf = float(boxes.conf[i])
                id_ = int(boxes.id[i])  # Real track ID from BoT-SORT
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                # Draw bounding box and label
                color = (0, 255, 0) if label == "students_not_cheating" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} | ID {id_}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Log and track cheating
                if label == "students_cheating":
                    strike_counts[id_] += 1
                    if strike_counts[id_] <= MAX_STRIKES:
                        cursor.execute('''
                            INSERT INTO detections (student_id, timestamp, label, strikes)
                            VALUES (?, ?, ?, ?)
                        ''', (str(id_), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, strike_counts[id_]))
                        conn.commit()
                    if strike_counts[id_] >= MAX_STRIKES:
                        cv2.putText(frame, "⚠️ MARKED CHEATING", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("BoT-SORT Cheat Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        cursor.execute("DELETE FROM detections")
        conn.commit()
        strike_counts.clear()
        print("🔄 All logs reset.")

cap.release()
cv2.destroyAllWindows()
conn.close()
