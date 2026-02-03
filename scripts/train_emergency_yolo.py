from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "data/dataset/data.yaml"

print("🚨 Training Emergency Vehicle YOLO Model...")

model = YOLO("yolov8n.pt")  # base model

model.train(
    data=str(DATASET),
    epochs=50,
    imgsz=640,
    batch=16
)

# Save best model
output_path = BASE_DIR / "models/emergency_detector.pt"
model.export(format="pt")
model.model.save(str(output_path))

print(f"🎉 Emergency Vehicle Model Saved → {output_path}")
