from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

# define the required steps
required_steps = {'step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6'}
detected_steps = set()
start_time = time.time()

cap = cv2.VideoCapture(0)

print("Hand wash detection started... Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # make predictions
    results = model.predict(frame, conf=0.45, verbose=False)[0]

    # process detections
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
    if start_time:
        elapsed_time = time.time() - start_timex
        if detected_steps == required_steps and elapsed_time >= 20 and not success_announced:
            print(f"Handwashing successful! All steps completed. Hands washed for {elapsed_time} seconds")
            break

    # exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting")
        break

cap.release()
cv2.destroyAllWindows()