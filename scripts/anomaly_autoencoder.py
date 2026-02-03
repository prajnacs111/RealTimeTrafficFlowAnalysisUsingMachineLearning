"""
Final Hybrid Anomaly Detection System WITH:
- Autoencoder reconstruction anomaly
- Spike-based sudden traffic jump detection
- Visualization timeline
- Logging for controller + dashboard
Compatible with TensorFlow/Keras 2.12
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "agg" / "traffic_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
LOG_FILE = BASE_DIR / "data" / "agg" / "anomaly_log.csv"

MODEL_DIR.mkdir(exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ---------------- FEATURES ----------------
FEATURES = ["total_vehicles", "temp", "humidity", "wind_speed", "rain_mm"]
df["traffic_smooth"] = df["total_vehicles"].rolling(window=10, min_periods=1).mean()

data = df[["traffic_smooth", "temp", "humidity", "wind_speed", "rain_mm"]].values

# ---------------- SCALING ----------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ---------------- WINDOWING ----------------
WINDOW = 20
X = []

for i in range(WINDOW, len(data_scaled)):
    X.append(data_scaled[i-WINDOW:i].flatten())

X = np.array(X)

# ---------------- TRAIN SPLIT ----------------
train_size = int(len(X) * 0.8)
train_data, test_data = X[:train_size], X[train_size:]

# ---------------- MODEL ----------------
model = Sequential([
    Dense(128, activation='relu', input_dim=X.shape[1]),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(X.shape[1], activation='linear')
])

model.compile(optimizer="adam", loss="mse")

print("\n🚀 Training Autoencoder...\n")
model.fit(train_data, train_data, epochs=30, batch_size=16, 
          validation_split=0.2, verbose=1)

# ---------------- RECONSTRUCTION ----------------
train_recon = model.predict(train_data, verbose=0)
test_recon = model.predict(test_data, verbose=0)

train_error = np.mean((train_data - train_recon)**2, axis=1)
test_error = np.mean((test_data - test_recon)**2, axis=1)

# ---------------- AUTOENCODER ANOMALY THRESHOLD ----------------
threshold = np.mean(train_error) + 4 * np.std(train_error)
print(f"\n📌 Autoencoder Threshold: {threshold:.6f}")

# ---------------- SPIKE DETECTION ----------------
df_test = df.iloc[WINDOW + train_size:].copy()
df_test["recon_error"] = test_error

df_test["spike"] = abs(df_test["total_vehicles"] - df_test["traffic_smooth"])
spike_threshold = df_test["spike"].mean() + 3 * df_test["spike"].std()

print(f"📌 Spike Threshold: {spike_threshold:.2f}")

# ---------------- FINAL ANOMALY DECISION ----------------
df_test["autoencoder_flag"] = df_test["recon_error"] > threshold
df_test["spike_flag"] = df_test["spike"] > spike_threshold

df_test["is_anomaly"] = df_test["autoencoder_flag"] | df_test["spike_flag"]

print("\n🔍 Anomaly Summary:")
print(df_test["is_anomaly"].value_counts())

# ---------------- SAVE LOG ----------------
df_test.to_csv(LOG_FILE, index=False)
print(f"\n📝 Anomaly log saved → {LOG_FILE}")

# ---------------- SAVE MODEL & SCALER ----------------
model.save(MODEL_DIR / "anomaly_autoencoder.keras")
np.save(MODEL_DIR / "anomaly_threshold.npy", threshold)
np.save(MODEL_DIR / "anomaly_spike_threshold.npy", spike_threshold)
joblib.dump(scaler, MODEL_DIR / "anomaly_scaler.pkl")

print("\n💾 Model + scaler saved.\n")

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(14,5))
plt.plot(df_test["timestamp"], df_test["recon_error"], label="Reconstruction Error")
plt.axhline(y=threshold, color="red", linestyle="--", label="Autoencoder Threshold")

# highlight anomalies
anom_times = df_test[df_test["is_anomaly"]]["timestamp"]
anom_vals = df_test[df_test["is_anomaly"]]["recon_error"]
plt.scatter(anom_times, anom_vals, color="orange", label="Detected Anomaly", s=40)

plt.xticks(rotation=45)
plt.title("🚨 Traffic Anomaly Detection Timeline")
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Visualization completed.")
