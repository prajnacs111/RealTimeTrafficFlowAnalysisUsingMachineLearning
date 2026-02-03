#test_yolo.py 
from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# -------- CONFIG --------
SOURCE = "data/sample.jpg"
model = YOLO('yolov8n.pt')

# Ensure logs/ folder exists
base_log_dir = "logs"
os.makedirs(base_log_dir, exist_ok=True)

# Create new run folder for this session (for images/frames)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(base_log_dir, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# ---- CSV toggle ----
ENABLE_CSV = True
if ENABLE_CSV:
    video_label = "image" if not SOURCE.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else os.path.splitext(os.path.basename(SOURCE))[0]
    csv_filename = os.path.join(base_log_dir, f"traffic_{video_label}_{timestamp}.csv")
    csv_fields = [
        "video_name","timestamp","frame","car","bus","truck","motorcycle","bicycle",
        "total_vehicles","person"
    ]
    with open(csv_filename, mode='w', newline='') as f:
        csv.writer(f).writerow(csv_fields)


# ---------------------------
# Draw detections (with class + confidence)
# ---------------------------
def draw_boxes(frame, boxes, names):
    if boxes is None:
        return frame

    colors = {
        "person": (255, 255, 255),      # white
        "car": (255, 255, 0),           # cyan
        "bus": (255, 0, 255),           # pink
        "truck": (0, 165, 255),         # orange
        "motorcycle": (0, 0, 255),      # blue
        "bicycle": (0, 0, 255),         # blue
    }

    xyxy = np.array(boxes.xyxy.cpu())
    conf = np.array(boxes.conf.cpu())
    cls = np.array(boxes.cls.cpu())

    for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_name = names[int(cl)]
        color = colors.get(class_name, (0, 255, 0))
        label = f"{class_name} {c:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# ---------------------------
def count_objects(boxes, names):
    vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]
    counts = {cls: 0 for cls in vehicle_classes}
    person_count = 0

    cls = np.array(boxes.cls.cpu())
    for cl in cls:
        class_name = names[int(cl)]
        if class_name in counts:
            counts[class_name] += 1
        elif class_name == "person":
            person_count += 1

    return counts, person_count


# ---------------------------
def run_on_image(path):
    img = cv2.imread(path)
    if img is None:
        print("❌ Failed to read image:", path)
        return

    results = model(path)
    res = results[0]
    out = draw_boxes(img, res.boxes, model.names)

    vehicle_counts, person_count = count_objects(res.boxes, model.names)
    total_vehicles = sum(vehicle_counts.values())
    print("Vehicle breakdown:", vehicle_counts, "| Persons:", person_count)

    # CSV logging (only one row for image)
    if ENABLE_CSV:
        with open(csv_filename, mode='a', newline='') as f:
            csv.writer(f).writerow([
                video_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                1, vehicle_counts['car'], vehicle_counts['bus'], vehicle_counts['truck'],
                vehicle_counts['motorcycle'], vehicle_counts['bicycle'],
                total_vehicles, person_count
            ])

    cv2.putText(out, f"Total Vehicles: {total_vehicles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, f"Persons: {person_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    save_path = os.path.join(run_dir, "image_output.jpg")
    cv2.imwrite(save_path, out)
    print(f"✅ Saved image to {save_path}")

    cv2.imshow('YOLOv8 - Image with Counts', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

        results = model(frame)
        res = results[0]
        out = draw_boxes(frame, res.boxes, model.names)

        vehicle_counts, person_count = count_objects(res.boxes, model.names)
        total_vehicles = sum(vehicle_counts.values())
        print("Vehicle breakdown:", vehicle_counts, "| Persons:", person_count)

        # CSV logging (append each frame)
        if ENABLE_CSV:
            with open(csv_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    video_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_id, vehicle_counts['car'], vehicle_counts['bus'], vehicle_counts['truck'],
                    vehicle_counts['motorcycle'], vehicle_counts['bicycle'],
                    total_vehicles, person_count
                ])

        cv2.putText(out, f"Total Vehicles: {total_vehicles}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(out, f"Persons: {person_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        save_path = os.path.join(run_dir, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(save_path, out)

        cv2.imshow('YOLOv8 - Video with Counts', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
if __name__ == "__main__":
    if not os.path.exists(SOURCE):
        print("⚠️ Put a sample image/video in data/ and set SOURCE variable in this file.")
    else:
        if SOURCE.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            run_on_video(SOURCE)
        else:
            run_on_image(SOURCE)
