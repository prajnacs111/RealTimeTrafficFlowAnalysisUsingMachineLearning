import numpy as np
import sounddevice as sd
import librosa
from keras.models import load_model
import time

MODEL_PATH = r"D:\real_time_traffic_flow\models\siren_detector.h5"
model = load_model(MODEL_PATH)

# Your labels (same order as folders)
LABELS = ["ambulance", "firetruck"]

SAMPLE_RATE = 22050
DURATION = 3  # seconds — same as training

def predict_siren(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.array([mfcc_processed]))[0]
    confidence = np.max(prediction)
    index = np.argmax(prediction)

    return LABELS[index], confidence


print("🎤 Listening for emergency sirens... (Press Ctrl + C to stop)\n")

try:
    while True:
        print("⏺ Recording...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        audio = audio.flatten()

        label, confidence = predict_siren(audio)

        if confidence > 0.70:
            print(f"🚨 Siren Detected: {label.upper()}  | Confidence: {confidence:.2f}")
        else:
            print(f"⚪ No emergency siren | Confidence: {confidence:.2f}")

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Detection stopped by user.")
