import cv2
import sqlite3
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Change if path is different

# Connect to SQLite DB
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT,
    confidence REAL,
    timestamp TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS attempts (
    class_name TEXT PRIMARY KEY,
    attempt_count INTEGER,
    last_detected TEXT
)
''')

conn.commit()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log detection
        cursor.execute("INSERT INTO logs (class_name, confidence, timestamp) VALUES (?, ?, ?)",
                       (class_name, confidence, timestamp))

        # Attempt tracking
        cursor.execute("SELECT attempt_count FROM attempts WHERE class_name = ?", (class_name,))
        row = cursor.fetchone()

        if row:
            attempt_count = row[0] + 1
            cursor.execute("UPDATE attempts SET attempt_count = ?, last_detected = ? WHERE class_name = ?",
                           (attempt_count, timestamp, class_name))
        else:
            attempt_count = 1
            cursor.execute("INSERT INTO attempts (class_name, attempt_count, last_detected) VALUES (?, ?, ?)",
                           (class_name, attempt_count, timestamp))

        conn.commit()

        # Draw on frame
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        if attempt_count >= 3:
            label = f"{class_name} - CHEATING ❌"
            color = (0, 0, 255)  # Red
        else:
            label = f"{class_name} | Attempt {attempt_count}/3"
            color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Student Detection - Anti-Cheat", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
conn.close()
