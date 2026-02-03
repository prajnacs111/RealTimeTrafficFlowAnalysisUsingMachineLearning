Project Title: Real Time Traffic Flow Analysis And Prediction System Using Dynamic Traffic Signal Control And Anomaly Detection.

Tools: VSCode, python 3.11.9

create python environment
python -m venv traf_env
.\traf_env\Scripts\activate


to activate python environment
D:\real_time_traffic_flow\traf_env\Scripts\activate



select python environment
crtl+shift+p
python3.11.9 (traf_env)  .\traf_env\Scripts\python.exe    (Recommended)

run      python test1_yolo.py


from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# -------- CONFIG --------
SOURCE = 0  # or "data/sample.mp4"
video_label = "webcam" if isinstance(SOURCE, int) else os.path.splitext(os.path.basename(SOURCE))[0]

# Ensure logs/ folder exists
os.makedirs("logs", exist_ok=True)

# CSV filename = logs/traffic_<video>_<datetime>.csv
timestamp_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
csv_filename = os.path.join("logs", f"traffic_{video_label}_{timestamp_now}.csv")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Write CSV header
csv_fields = [
    "video_name","timestamp","frame","fps",
    "car","bus","truck","motorcycle","bicycle",
    "total_vehicles","person"
]
with open(csv_filename, mode='w', newline='') as f:
    csv.writer(f).writerow(csv_fields)


# ---------------------------
# Draw detections
# ---------------------------
def draw_boxes(frame, boxes, names):
    if boxes is None:
        return frame
    colors = {
        "person": (255, 255, 255),      # white
        "car": (255, 255, 0),           # cyan
        "bus": (255, 0, 255),            # pink
        "truck": (255, 165, 0),         # orange
        "motorcycle": (0, 0, 255),      # blue
        "bicycle": (0, 0, 255)          # blue
    }

    xyxy = np.array(boxes.xyxy.cpu())
    conf = np.array(boxes.conf.cpu())
    cls = np.array(boxes.cls.cpu())

    for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_name = names[int(cl)]
        color = colors.get(class_name, (0, 255, 0))
        label = f"{c:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


# ---------------------------
# Count vehicles & persons
# ---------------------------
def count_objects(boxes, names):
    vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]
    counts = {cls: 0 for cls in vehicle_classes}
    person_count = 0
    cls = np.array(boxes.cls.cpu())
    for cl in cls:
        cname = names[int(cl)]
        if cname in counts:
            counts[cname] += 1
        elif cname == "person":
            person_count += 1
    return counts, person_count


# ---------------------------
# Run detection on VIDEO
# ---------------------------
def run_on_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("❌ Failed to open video:", path)
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        start_time = cv2.getTickCount()
        results = model(frame)
        end_time = cv2.getTickCount()

        fps = cv2.getTickFrequency() / (end_time - start_time)
        res = results[0]

        out = draw_boxes(frame, res.boxes, model.names)

        vehicle_counts, person_count = count_objects(res.boxes, model.names)
        total_vehicles = sum(vehicle_counts.values())

        # --- Terminal ---
        print(f"Frame {frame_id} | Vehicles: {vehicle_counts} | Persons: {person_count} | FPS: {fps:.2f}")

        # --- Overlay ---
        cv2.putText(out, f"Total Vehicles: {total_vehicles}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   # red
        cv2.putText(out, f"Persons: {person_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # white
        cv2.putText(out, f"FPS: {fps:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # --- CSV ---
        with open(csv_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([
                video_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_id, round(fps,2),
                vehicle_counts['car'], vehicle_counts['bus'], vehicle_counts['truck'],
                vehicle_counts['motorcycle'], vehicle_counts['bicycle'],
                total_vehicles, person_count
            ])

        cv2.imshow('YOLOv8 - Video with Counts', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    run_on_video(SOURCE)

