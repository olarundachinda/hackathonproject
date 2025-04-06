from ultralytics import YOLO
import cv2
import time
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define the required steps
required_steps = {'step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6'}
detected_steps = set()
label_frame_counts = defaultdict(int)
frame_threshold = 3
start_time = None
success_announced = False

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

print("Hand wash detection started... Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions
    results = model.predict([frame], conf=0.20, verbose=False)[0]

    # Process detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if label == "step_1" and conf >= 0.20:
            label_frame_counts[label] += 1
        elif label in required_steps and conf >= 0.45:
            label_frame_counts[label] += 1
        else:
            continue

        if label_frame_counts[label] == frame_threshold:
            if label not in detected_steps:
                print(f"Detected: {label} ({conf:.2f})")
                detected_steps.add(label)

            if start_time is None:
                start_time = time.time()

    # show the frame
    cv2.imshow("Handwash Detection", frame)

    # check for success
    if start_time is not None:
        elapsed_time = time.time() - start_time
        if detected_steps == required_steps and elapsed_time >= 20 and not success_announced:
            print(f"Handwashing successful! All steps completed. Hands washed for {elapsed_time} seconds")
            success_announced = True
            break

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting")
        break

cap.release()
cv2.destroyAllWindows()