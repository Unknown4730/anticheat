from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")  # Update path as needed

# Initialize webcam
cap = cv2.VideoCapture(0)

# Cheat attempt tracker
cheat_counts = defaultdict(int)
max_cheat_limit = 3
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(frame, conf=0.5)

    # Parse results
    annotated_frame = results[0].plot()
    detections = results[0].boxes

    if detections is not None:
        for i, box in enumerate(detections):
            cls_id = int(box.cls[0])  # Class index
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            if cls_id == 1:  # 'students_cheating' class
                obj_id = f"{x1}-{y1}-{x2}-{y2}"  # crude ID from box
                cheat_counts[obj_id] += 1

                # Draw red warning if > 2
                if cheat_counts[obj_id] >= max_cheat_limit:
                    cv2.putText(annotated_frame, "Cheater ⚠️", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, f"Strike {cheat_counts[obj_id]}/3", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("Cheating Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
