# track_yolo_sort_logger.py
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sort import Sort

# ---------------- CONFIGURATION ----------------
VIDEO_DIR = Path(r"D:\real_time_traffic_flow\data\videos")
LOG_DIR = Path(r"D:\real_time_traffic_flow\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = r"D:\real_time_traffic_flow\yolov8n.pt"
FRAME_SKIP = 5
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.3

print("🚦 YOLOv8 + SORT batch tracker started!")
print(f"📁 Searching for videos in: {VIDEO_DIR}\n")

# ---------------- LOAD YOLO MODEL ----------------
model = YOLO(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# ---------------- INIT SORT TRACKER ----------------
tracker = Sort(max_age=15, min_hits=3, iou_threshold=IOU_THRESHOLD)

# ---------------- CLASS MAP ----------------
VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ---------------- TIMESTAMP EXTRACTOR ----------------
def extract_video_time(video_name):
    """
    Extract timestamp from video filename format:
    Bellevue_Bellevue_NE8th_2017-09-10_18-08-23.mp4
    """
    try:
        date_part = video_name.split("_")[-2]
        time_part = video_name.split("_")[-1].replace(".mp4", "")
        return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H-%M-%S")
    except:
        print(f"⚠️ Timestamp not found in filename → using current time instead.")
        return datetime.now()

# ---------------- PROCESS EACH VIDEO ----------------
for video_path in VIDEO_DIR.glob("*.*"):
    if video_path.suffix.lower() not in [".mov", ".mp4", ".avi"]:
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open {video_path.name}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_minutes = (total_frames / fps) / 60 if fps else 0

    print(f"🎥 Processing: {video_path.name} | Duration: {duration_minutes:.2f} min | FPS: {fps:.1f}")

    # Extract correct start timestamp from filename
    video_start_time = extract_video_time(video_path.name)

    results_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # Calculate real timestamp based on frame position
        elapsed_seconds = frame_idx / fps
        timestamp = (video_start_time + timedelta(seconds=elapsed_seconds)).strftime("%Y-%m-%d %H:%M:%S")

        # YOLO prediction
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device="cpu")

        boxes = results[0].boxes
        if boxes is None or len(boxes.xyxy) == 0:
            frame_idx += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        det_for_tracker = np.hstack((xyxy, conf.reshape(-1, 1)))
        tracked_objects = tracker.update(det_for_tracker)

        frame_counts = {"timestamp": timestamp}

        for name in VEHICLE_CLASSES.values():
            frame_counts[name] = 0

        for c in cls:
            cls_id = int(c)
            label = VEHICLE_CLASSES.get(cls_id)
            if label:
                frame_counts[label] += 1

        frame_counts["total_vehicles"] = sum(
            v for k, v in frame_counts.items() if isinstance(v, (int, float)) and k != "timestamp"
        )

        results_data.append(frame_counts)
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(results_data)
    out_name = f"{video_path.stem}_LOG.csv"
    df.to_csv(LOG_DIR / out_name, index=False)
    print(f"✅ Saved CSV → {LOG_DIR/out_name}\n")

print("🎉 YOLOv8 + SORT tracking completed!")
print("➡ Run merge_logs.py next.")
