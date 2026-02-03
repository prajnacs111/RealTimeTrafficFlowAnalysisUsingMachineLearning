#test1_yolo.property
from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
import sounddevice as sd
import librosa
import threading
from keras.models import load_model
from glob import glob
from datetime import datetime

# ---------------------- CONFIG ----------------------
SOURCE = 0  
SAMPLE_RATE = 22050
AUDIO_DURATION = 3
SIREN_THRESHOLD = 0.80

LABELS = ["ambulance", "firetruck"]

SIREN_MODEL_PATH = r"D:\real_time_traffic_flow\models\siren_detector.h5"
SEARCH_PATTERN = r"D:\real_time_traffic_flow\runs\detect\**\weights\best.pt"

found_models = glob(SEARCH_PATTERN, recursive=True)
if not found_models:
    raise FileNotFoundError("\n❌ No emergency YOLO model found.\n")

YOLO_EMERGENCY_MODEL = found_models[-1]
YOLO_GENERAL_MODEL = "yolov8n.pt"

print(f"🚨 Emergency YOLO Model: {YOLO_EMERGENCY_MODEL}")
print(f"🚗 General YOLO Model: {YOLO_GENERAL_MODEL}")
print(f"🔊 Siren Model Loaded: {SIREN_MODEL_PATH}")

# Load models
model_audio = load_model(SIREN_MODEL_PATH)
model_general = YOLO(YOLO_GENERAL_MODEL)
model_emergency = YOLO(YOLO_EMERGENCY_MODEL)

siren_detected = False

# ---------------------- LOGGING ----------------------
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
csv_file = f"logs/traffic_log_{timestamp}.csv"

with open(csv_file, "w", newline="") as f:
    csv.writer(f).writerow([
        "timestamp","frame","fps",
        "person","car","bus","truck","motorcycle","bicycle",
        "emergency_vehicle","audio_siren_detected","total_vehicle_count"
    ])

# ---------------------- COLORS ----------------------
COLORS = {
    "person": (180,180,180),
    "car": (0, 255, 255),
    "bus": (255, 0, 255),
    "truck": (0, 165, 255),
    "motorcycle": (0, 255, 0),
    "bicycle": (255, 255, 0),
    "emergency_vehicle": (0, 0, 255),
}

vehicle_classes = ["car","bus","truck","motorcycle","bicycle","person"]

# ---------------------- AUDIO THREAD ----------------------
def listen_siren():
    global siren_detected
    while True:
        audio = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        result = model_audio.predict(np.array([mfcc]))[0]
        confidence = max(result)

        if confidence > SIREN_THRESHOLD:
            siren_detected = True
            print(f"🔊 AUDIO: {LABELS[np.argmax(result)].upper()} | CONF={confidence:.2f}")
        else:
            siren_detected = False


# ---------------------- FUSION + REMOVE DUPLICATES ----------------------
def remove_duplicate_boxes(boxes, iou_threshold=0.5):
    filtered = []

    def iou(a, b):
        x1, y1, x2, y2 = a
        xx1, yy1, xx2, yy2 = b

        inter_x1 = max(x1, xx1)
        inter_y1 = max(y1, yy1)
        inter_x2 = min(x2, xx2)
        inter_y2 = min(y2, yy2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        a_area = (x2 - x1) * (y2 - y1)
        b_area = (xx2 - xx1) * (yy2 - yy1)

        return inter_area / max(a_area + b_area - inter_area, 1)

    for box in boxes:
        if not any(iou(box[:4], f[:4]) > iou_threshold and box[4] == f[4] for f in filtered):
            filtered.append(box)

    return filtered


def fuse_and_draw(frame, general, emergency):
    final_boxes = []

    for r in general:
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            cls = r.names[int(b.cls)]
            conf = float(b.conf)
            if cls in vehicle_classes:
                final_boxes.append((x1,y1,x2,y2,cls,conf))

    for r in emergency:
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            conf = float(b.conf)
            final_boxes.append((x1,y1,x2,y2,"emergency_vehicle",conf))

    final_boxes = remove_duplicate_boxes(final_boxes)

    for (x1,y1,x2,y2,label,conf) in final_boxes:
        cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS[label],2)
        cv2.putText(frame,f"{label} ({conf:.2f})",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLORS[label],2)

    return frame, final_boxes


# ---------------------- COUNT ----------------------
def summarize(boxes):
    counts = {cls:0 for cls in vehicle_classes}
    counts["emergency_vehicle"] = 0

    for (_,_,_,_,cls,_) in boxes:
        counts[cls] += 1
        
    return counts, sum(counts.values())


# ---------------------- MAIN ----------------------
def run():
    threading.Thread(target=listen_siren, daemon=True).start()
    cap = cv2.VideoCapture(SOURCE)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_id += 1
        start = cv2.getTickCount()

        r1 = model_general(frame)
        r2 = model_emergency(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)

        frame, detections = fuse_and_draw(frame, r1, r2)
        counts, total = summarize(detections)

        cv2.putText(frame,f"Vehicles: {total}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(frame,f"FPS: {fps:.2f}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

        alert = counts["emergency_vehicle"] > 0 or siren_detected
        if alert:
            print("🚨 EMERGENCY MODE ENABLED (Video or Audio Triggered)")

        with open(csv_file,"a",newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), frame_id, round(fps,2),
                counts["person"], counts["car"], counts["bus"], counts["truck"], 
                counts["motorcycle"], counts["bicycle"],
                counts["emergency_vehicle"], siren_detected, total
            ])

        cv2.imshow("🚦 Smart Traffic AI System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📁 Log saved at: {csv_file}")


if __name__ == "__main__":
    run()
