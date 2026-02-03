"""
Final Siren Detection Training Script
- MFCC Extraction
- Neural Network Classifier
- Saves Model + Scaler + Class Map
Compatible with TensorFlow/Keras 2.12
"""

import os
import numpy as np
import librosa
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "siren_sounds" / "sounds"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file):
    audio, sr = librosa.load(file, duration=3)  # Normalize duration
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []
class_names = sorted(os.listdir(DATASET_PATH))

print("\n🎧 Extracting audio features...\n")

for label_idx, folder in enumerate(class_names):
    folder_path = DATASET_PATH / folder

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            fpath = folder_path / file
            print("📌 Processing:", fpath)

            features.append(extract_features(fpath))
            labels.append(label_idx)

X = np.array(features)
y = to_categorical(labels)
print("\n✔ Features shape:", X.shape)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Training Siren Detection Model...\n")
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test))

# ---------------- EVALUATION ----------------
print("\n📊 Model Evaluation:\n")
preds = model.predict(X_test)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test, axis=1)

print(classification_report(true_classes, pred_classes, target_names=class_names))
print("Confusion Matrix:\n", confusion_matrix(true_classes, pred_classes))

# ---------------- SAVE ARTIFACTS ----------------
model.save(MODEL_DIR / "siren_detector.keras")
joblib.dump(scaler, MODEL_DIR / "siren_scaler.pkl")

with open(MODEL_DIR / "siren_labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n🎉 Siren Model Training Complete!")
print("📁 Saved:")
print("- Model → siren_detector.keras")
print("- Scaler → siren_scaler.pkl")
print("- Labels → siren_labels.txt")
