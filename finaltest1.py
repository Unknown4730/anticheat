import cv2
import sqlite3
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import random  # Mocking LSTM prediction

# -------------------- DATABASE SETUP --------------------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cheat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT,
            timestamp TEXT,
            attempts INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def log_attempt(person_id):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT attempts FROM cheat_logs WHERE person_id = ?", (person_id,))
    row = cursor.fetchone()

    if row:
        attempts = row[0] + 1
        cursor.execute("UPDATE cheat_logs SET attempts = ?, timestamp = ? WHERE person_id = ?",
                       (attempts, datetime.now().isoformat(), person_id))
    else:
        attempts = 1
        cursor.execute("INSERT INTO cheat_logs (person_id, timestamp, attempts) VALUES (?, ?, ?)",
                       (person_id, datetime.now().isoformat(), attempts))

    conn.commit()
    conn.close()
    return attempts

def reset_logs():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cheat_logs")
    conn.commit()
    conn.close()
    print("[INFO] Logs reset manually.")

# -------------------- MOCK LSTM PREDICTION --------------------
def predict_with_lstm(features):
    # Replace this logic with your LSTM model
    return random.choice([True, False])  # Randomly detect cheating

# -------------------- MAIN DETECTION LOOP --------------------
def run_anti_cheat():
    cap = cv2.VideoCapture(0)
    model = YOLO(r"P:\bhauji pics\runs\detect\train\weights\best.pt")

    print("[INFO] Starting webcam... Press 'r' to reset logs, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Assign a mock person ID based on coordinates (or use actual tracker ID)
            person_id = f"{x1}_{y1}"

            # Extract features for LSTM
            features = [x1, y1, x2 - x1, y2 - y1, cls, conf]
            cheat_pred = predict_with_lstm(features)

            # Draw bounding box
            color = (0, 255, 0)
            label = "Normal"
            if cheat_pred:
                attempts = log_attempt(person_id)
                label = f"Cheat {attempts}/3"
                color = (0, 0, 255)
                if attempts >= 3:
                    label = "Cheater 🚫"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Anti-Cheat System", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_logs()

    cap.release()
    cv2.destroyAllWindows()

# -------------------- RUN SCRIPT --------------------
if __name__ == "__main__":
    init_db()
    run_anti_cheat()
