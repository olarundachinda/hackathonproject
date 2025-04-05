from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")
required_steps = {"step1_left", "step1_right", "step2_left", "step2_right", "step3_left", "step3_right"}
completed_steps = set()
start_time = None

# Start real-time tracking from webcam
for result in model.track(source=1, show=True, tracker="botsort.yaml", stream=True):
    # Get detections
    names = result.names  # class index to label
    boxes = result.boxes
    if boxes is None:
        continue

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        
        if conf > 0.45 and label in required_steps:
            completed_steps.add(label)
            print(f"✅ Detected: {label} with confidence {conf}")

    # Start timer once any valid step is detected
    if completed_steps and start_time is None:
        start_time = time.time()

    # Check if 20 seconds passed and all steps are done
    if start_time:
        elapsed = time.time() - start_time
        if elapsed >= 20 and completed_steps == required_steps:
            print("🎉 Successful hand wash detected!")
            break
