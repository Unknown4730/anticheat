import cv2
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
model = YOLO('yolov8n.pt')

WINDOW_TITLE = "Anti-Cheat Live Feed"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video source")

frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 2 != 0:  # Skip every other frame for speed
            continue
        resized_frame = cv2.resize(frame, (320, 320))
        results = model(resized_frame)
        annotated_frame = results[0].plot()
        cv2.imshow(WINDOW_TITLE, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
