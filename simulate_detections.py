"""
Simulate detection events to verify strike/flagging logic without a camera.

This simple script replays a list of fake detection events (id,label) across frames
and uses the Detector's recording logic by calling the DB recording function directly.
"""
import time
import os
from datetime import datetime
import sqlite3

RUNTIME_DIR = "runtime"
DB_PATH = os.path.join(RUNTIME_DIR, "cheat_logs.db")
os.makedirs(RUNTIME_DIR, exist_ok=True)

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
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

def simulate():
    conn = init_db()
    # Simulated timeline: each entry is list of detections in that frame: (id,label)
    timeline = [
        [(1,"students_not_cheating"), (2,"students_not_cheating")],
        [(1,"students_cheating"), (2,"students_not_cheating")],
        [(1,"students_cheating"), (2,"students_cheating")],
        [(1,"students_cheating"), (2,"students_cheating")],
    ]
    strike_counts = {}
    for frame in timeline:
        for sid, label in frame:
            if label == "students_cheating":
                strike_counts[sid] = strike_counts.get(sid, 0) + 1
                record_detection(conn, sid, label, strike_counts[sid])
                print(f"Recorded cheating for ID {sid}. strikes={strike_counts[sid]}")
        time.sleep(0.5)
    print("Simulation complete. DB saved to", DB_PATH)

if __name__ == "__main__":
    simulate()
