from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define the required steps
required_steps = {'step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6'}
detected_steps = set()
start_time = time.time()
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
    results = model.predict(frame, conf=0.45, verbose=False)[0]

    # Process detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if conf >= 0.45 and label in required_steps:
            if label not in detected_steps:
                print(f"Detected: {label} ({conf:.2f})")
            detected_steps.add(label)

    # show the frame
    cv2.imshow("Handwash Detection", frame)

    # check for success
    elapsed_time = time.time() - start_time
    if detected_steps == required_steps and elapsed_time >= 20 and not success_announced:
        print(f"Handwashing successful! All steps completed. Hands washed for {elapsed_time} seconds")
        success_announced = True

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting")
        break

cap.release()
cv2.destroyAllWindows()