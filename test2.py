from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load model
model = YOLO("runs/detect/train/weights/best.pt")  # 👈 update path

# Start webcam
cap = cv2.VideoCapture(0)

# Dict to count cheating attempts
cheat_counts = defaultdict(int)
max_strikes = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Loop through detections
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  # class index
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
        conf = float(box.conf[0])

        if cls_id == 1:  # students_cheating
            # crude ID using box coordinates
            obj_id = f"{x1}-{y1}-{x2}-{y2}"
            cheat_counts[obj_id] += 1
            strikes = cheat_counts[obj_id]

            if strikes >= max_strikes:
                cv2.putText(annotated_frame, "CHEATER ⚠️", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(annotated_frame, f"Strike {strikes}/3", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Live YOLO Cheating Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
