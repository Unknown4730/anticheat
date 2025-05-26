import cv2
import sqlite3
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO

# --- SETTINGS ---
CONFIDENCE_THRESHOLD = 0.5
MAX_STRIKES = 3
SMOOTHING_HISTORY = 5
MODEL_PATH = "runs/detect/train/weights/best.pt"  # Update path if needed

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

# --- INIT YOLO MODEL ---
model = YOLO(MODEL_PATH)
model.fuse()

# --- STATE TRACKERS ---
strike_counts = defaultdict(int)
box_history = defaultdict(lambda: deque(maxlen=SMOOTHING_HISTORY))

# --- CAPTURE VIDEO ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]

    for box in results.boxes:
        if box.conf < CONFIDENCE_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Fake ID (since no tracker): Use bounding box midpoint as a weak ID
        track_id = f"{(x1 + x2) // 2}_{(y1 + y2) // 2}"

        # Store smoothed box
        box_history[track_id].append((x1, y1, x2, y2))
        sx1, sy1, sx2, sy2 = map(lambda x: int(sum(x)/len(x)), zip(*box_history[track_id]))

        # Draw box
        color = (0, 255, 0) if label == "students_not_cheating" else (0, 0, 255)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.putText(frame, f"{label} | {track_id}", (sx1, sy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Log and strike count
        if label == "students_cheating":
            strike_counts[track_id] += 1
            if strike_counts[track_id] <= MAX_STRIKES:
                cursor.execute('''
                    INSERT INTO detections (student_id, timestamp, label, strikes)
                    VALUES (?, ?, ?, ?)
                ''', (track_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, strike_counts[track_id]))
                conn.commit()
            if strike_counts[track_id] >= MAX_STRIKES:
                cv2.putText(frame, "⚠️ Marked Cheating", (sx1, sy2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Smoothed Cheating Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('r'):
        cursor.execute("DELETE FROM detections")
        conn.commit()
        strike_counts.clear()
        print("🔄 All logs reset.")

cap.release()
cv2.destroyAllWindows()
conn.close()
