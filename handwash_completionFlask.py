from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import time
import logging


model_path = "yolov8n.pt"
required_steps = {'step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6'}
confidence_threshold = 0.45
success_time = 20

app = Flask(__name__)
model = YOLO(model_path)

logging.basicConfig(level=logging.INFO)

def gen_frames():
    cap = cv2.VideoCapture(0)
    detected_steps = set()
    start_time = None
    success_announced = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=confidence_threshold, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if conf >= 0.45 and label in required_steps:
                if label not in detected_steps:
                    print(f"Detected: {label} ({conf:.2f})")
                detected_steps.add(label)
                if start_time is None:
                    start_time = time.time()

        # Check success
        if start_time:
            elapsed_time = time.time() - start_time
            if detected_steps == required_steps and elapsed_time >= 20 and not success_announced:
                cv2.putText(frame, f"Handwashing successful! ({int(elapsed_time)}s)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                success_announced = True
            else:
                cv2.putText(frame, f"Steps: {len(detected_steps)}/6 | Time: {int(elapsed_time)}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
