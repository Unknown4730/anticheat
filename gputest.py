import cv2
import sqlite3
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# --- Settings ---
MODEL_PATH = "best.pt"
TRACKER_CONFIG = "botsort.yaml"
CONFIDENCE_THRESHOLD = 0.5
MAX_STRIKES = 3

# --- Load model
model = YOLO(MODEL_PATH)

# --- SQLite DB setup
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

# --- Strike counter
strike_counts = defaultdict(int)

# --- Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("[INFO] Running with BoT-SORT + Re-ID... Press 'q' to quit or 'r' to reset logs.")

while True:
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
                        cv2.putText(frame, f"⚠️ Cheating Detected", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Anti-Cheat Detection (BoT-SORT + ReID)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        strike_counts.clear()
        cursor.execute("DELETE FROM detections")
        conn.commit()
        print("[INFO] Logs reset!")

cap.release()
cv2.destroyAllWindows()
conn.close()
