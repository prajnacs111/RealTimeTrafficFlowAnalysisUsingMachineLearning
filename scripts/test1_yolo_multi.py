# test1_yolo_multi.py
"""
YOLOv8 + Emergency Vehicle + Siren Detection (Per-Lane Logger)

- Detects vehicles + emergency vehicles with YOLOv8
- Detects siren sound using keras audio model
- Logs per-lane counts to CSV for LSTM + controller
"""

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
from pathlib import Path
import joblib
import argparse

# ==========================================================
# ARGUMENTS
# ==========================================================
parser = argparse.ArgumentParser(description="YOLO + Siren lane capture")
parser.add_argument("--lane", type=int, required=True, help="Lane ID (1–4)")
parser.add_argument("--source", required=True, help="Camera index or video path")
parser.add_argument("--audio", action="store_true", help="Enable siren detection")
args = parser.parse_args()

LANE_ID = args.lane
SOURCE = int(args.source) if args.source.isdigit() else args.source
USE_AUDIO = args.audio

print(f"\n🚦 Starting YOLO for Lane {LANE_ID}")
print(f"🎥 Source: {SOURCE}")
print(f"🎤 Audio: {'ON' if USE_AUDIO else 'OFF'}")

# ==========================================================
# PATHS / CONFIG
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs" / "detect"
LOGS_ROOT = BASE_DIR / "scripts" / "logs"  # per-lane logs

SAMPLE_RATE = 22050
AUDIO_DURATION = 2          # seconds
CONF_THRESHOLD = 0.45
SIREN_THRESHOLD = 0.80

SIREN_MODEL_PATH = MODELS_DIR / "siren_detector.keras"
SIREN_SCALER_PATH = MODELS_DIR / "siren_scaler.pkl"
SIREN_LABELS_PATH = MODELS_DIR / "siren_labels.txt"

SEARCH_PATTERN = str(RUNS_DIR / "**" / "weights" / "best.pt")

# ==========================================================
# YOLO MODELS
# ==========================================================
print("\n📥 Loading YOLO models...")

found_models = glob(SEARCH_PATTERN, recursive=True)
if not found_models:
    raise FileNotFoundError("\n❌ No emergency YOLO model found. Train YOLO first.")

YOLO_GENERAL = "yolov8n.pt"
YOLO_EMERGENCY = found_models[-1]

print("✅ General Model :", YOLO_GENERAL)
print("✅ Emergency Model:", YOLO_EMERGENCY)

model_general = YOLO(YOLO_GENERAL)
model_emergency = YOLO(YOLO_EMERGENCY)

# ==========================================================
# AUDIO / SIREN MODEL
# ==========================================================
siren_detected = False
siren_scaler = None
siren_class_index = None

if USE_AUDIO:
    try:
        print(f"\n📥 Loading Siren Model: {SIREN_MODEL_PATH}")
        audio_model = load_model(SIREN_MODEL_PATH, compile=True)

        # Load scaler (for MFCC normalization)
        if SIREN_SCALER_PATH.exists():
            siren_scaler = joblib.load(SIREN_SCALER_PATH)
            print(f"✅ Siren scaler loaded: {SIREN_SCALER_PATH}")
        else:
            print("⚠ No siren_scaler.pkl found. Using raw MFCC (may reduce accuracy).")

        # Load label mapping to find 'siren' index (if file exists)
        if SIREN_LABELS_PATH.exists():
            with open(SIREN_LABELS_PATH, "r") as f:
                siren_labels = [ln.strip() for ln in f if ln.strip()]
            try:
                siren_class_index = siren_labels.index("siren")
                print(f"✅ Siren class index: {siren_class_index} (labels: {siren_labels})")
            except ValueError:
                print("⚠ 'siren' not found in siren_labels.txt. Using max probability.")
        else:
            print("⚠ siren_labels.txt not found. Using max probability.")

    except Exception as e:
        print(f"❌ Error loading siren model: {e}")
        print("🔇 Disabling audio detection.")
        USE_AUDIO = False
else:
    print("\n🔇 Siren Detection Disabled")

# ==========================================================
# LOGGING SETUP
# ==========================================================
lane_log_dir = LOGS_ROOT / f"lane_{LANE_ID}"
lane_log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
csv_file = lane_log_dir / f"lane{LANE_ID}_{timestamp}.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "lane", "frame", "fps",
        "person", "car", "bus", "truck", "motorcycle", "bicycle",
        "emergency_vehicle", "audio_siren_detected", "total_vehicle_count"
    ])

print(f"\n📄 Logging to: {csv_file}")

# ==========================================================
# AUDIO THREAD
# ==========================================================
def listen_siren():
    """Continuously listen for sirens on a background thread."""
    global siren_detected

    while True:
        audio = sd.rec(
            int(AUDIO_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio.flatten(),
            sr=SAMPLE_RATE,
            n_mfcc=40
        )
        features = np.mean(mfcc.T, axis=0)

        # Scale if scaler is available
        if siren_scaler is not None:
            features = siren_scaler.transform([features])[0]

        # Predict probabilities
        prediction = audio_model.predict(np.array([features]), verbose=0)[0]

        if siren_class_index is not None:
            siren_prob = float(prediction[siren_class_index])
        else:
            siren_prob = float(np.max(prediction))

        siren_detected = siren_prob > SIREN_THRESHOLD

        # Optional debug: print when detected
        if siren_detected:
            print(f"🚨 Siren detected! prob={siren_prob:.2f}")


if USE_AUDIO:
    threading.Thread(target=listen_siren, daemon=True).start()

# ==========================================================
# YOLO DETECTION
# ==========================================================
vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle", "person"]
COLORS = {
    "person": (180, 180, 180),
    "car": (0, 255, 255),
    "bus": (255, 0, 255),
    "truck": (0, 165, 255),
    "motorcycle": (0, 255, 0),
    "bicycle": (255, 255, 0),
    "emergency_vehicle": (0, 0, 255),
}

def process_frame(frame):
    """Run YOLO general + emergency models and return boxes."""
    boxes = []

    # General model
    results_gen = model_general(frame)
    for r in results_gen:
        for b in r.boxes:
            conf = float(b.conf)
            cls = r.names[int(b.cls)]
            if conf > CONF_THRESHOLD and cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append((x1, y1, x2, y2, cls, conf))

    # Emergency model
    results_emg = model_emergency(frame)
    for r in results_emg:
        for b in r.boxes:
            conf = float(b.conf)
            if conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append((x1, y1, x2, y2, "emergency_vehicle", conf))

    return boxes

# ==========================================================
# MAIN LOOP
# ==========================================================
def run():
    cap = cv2.VideoCapture(SOURCE)
    frame_id = 0

    if not cap.isOpened():
        print(f"❌ Could not open source: {SOURCE}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n❌ Video/Camera stream ended.")
            break

        frame_id += 1
        t0 = cv2.getTickCount()

        detections = process_frame(frame)
        fps = round(cv2.getTickFrequency() / (cv2.getTickCount() - t0), 2)

        # Count per class
        counts = {cls: 0 for cls in vehicle_classes}
        counts["emergency_vehicle"] = 0

        for x1, y1, x2, y2, label, conf in detections:
            counts[label] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[label], 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLORS[label],
                2,
            )

        total = sum(counts.values())

        # Log to CSV
        with open(csv_file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                LANE_ID,
                frame_id,
                fps,
                counts["person"],
                counts["car"],
                counts["bus"],
                counts["truck"],
                counts["motorcycle"],
                counts["bicycle"],
                counts["emergency_vehicle"],
                siren_detected,
                total,
            ])

        cv2.putText(
            frame,
            f"Lane {LANE_ID} | {total} vehicles",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        

        cv2.imshow(f"Lane {LANE_ID}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
