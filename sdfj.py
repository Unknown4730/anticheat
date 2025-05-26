from ultralytics import YOLO
import cv2

# Load your trained model (adjust path if needed)
model = YOLO("runs/detect/train/weights/best.pt")

# Use webcam (change 0 to a video file path if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model.predict(source=frame, show=True, conf=0.5)

    # Optional: Draw results directly on the frame
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("Cheating Detector", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
