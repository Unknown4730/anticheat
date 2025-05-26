import cv2
import sqlite3
from datetime import datetime
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Update if needed

# Create (or connect to) a SQLite database
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

# Create a table for logs if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT,
    confidence REAL,
    timestamp TEXT
)
''')
conn.commit()

# Start webcam feed
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

        # Log to database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO logs (class_name, confidence, timestamp) VALUES (?, ?, ?)",
                       (class_name, confidence, timestamp))
        conn.commit()

        # Draw box on frame
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Detection + Logging", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()
