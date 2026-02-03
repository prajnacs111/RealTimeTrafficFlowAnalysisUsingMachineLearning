"""
Final LSTM Traffic Forecast Model WITH Future Prediction
Runs full 70 epochs (no early stopping)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from pathlib import Path
import joblib

# ---------------- Load Cleaned Dataset ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "agg" / "traffic_cleaned.csv"

print("📥 Loading dataset...")
df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ---------------- Feature Columns ----------------
FEATURES = ["total_vehicles", "temp", "humidity", "wind_speed", "rain_mm"]
print("📌 Using features:", FEATURES)

data = df[FEATURES].values

# ---------------- Scaling ----------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler for future predictions
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(scaler, MODEL_DIR / "lstm_scaler.pkl")
print("💾 Saved scaler → models/lstm_scaler.pkl")

# ---------------- Build Training Windows ----------------
LOOKBACK = 30
X, y = [], []

for i in range(LOOKBACK, len(data_scaled)):
    X.append(data_scaled[i - LOOKBACK:i])
    y.append(data_scaled[i][0])

X, y = np.array(X), np.array(y)

# ---------------- Train/Test Split ----------------
SPLIT = int(0.8 * len(X))
X_train, X_test = X[:SPLIT], X[SPLIT:]
y_train, y_test = y[:SPLIT], y[SPLIT:]

time_train = df["timestamp"].iloc[LOOKBACK:LOOKBACK+len(y_train)]
time_test = df["timestamp"].iloc[LOOKBACK+len(y_train):LOOKBACK+len(y)]

print(f"🟢 TRAIN: {len(X_train)} samples")
print(f"🟡 TEST: {len(X_test)} samples")

# ---------------- LSTM Model ----------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOKBACK, X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("\n🚀 Training model (full 70 epochs)...\n")
model.fit(X_train, y_train, epochs=70, batch_size=16,
          validation_data=(X_test, y_test), verbose=1)

# ---------------- Predict Train/Test ----------------
train_pred_scaled = model.predict(X_train, verbose=0)
test_pred_scaled = model.predict(X_test, verbose=0)

# ---------------- Reverse Scale ----------------
def inverse(vals):
    dummy = np.zeros((len(vals), data.shape[1]))
    dummy[:, 0] = vals.reshape(-1)
    return scaler.inverse_transform(dummy)[:, 0]

train_pred = inverse(train_pred_scaled)
test_pred = inverse(test_pred_scaled)
y_train_real = inverse(y_train)
y_test_real = inverse(y_test)

# ---------------- Metrics ----------------
train_rmse = np.sqrt(mean_squared_error(y_train_real, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
train_r2 = r2_score(y_train_real, train_pred)
test_r2 = r2_score(y_test_real, test_pred)

print("\n📈 PERFORMANCE:")
print(f"🟢 Train: RMSE={train_rmse:.2f} | R²={train_r2:.3f}")
print(f"🟡 Test : RMSE={test_rmse:.2f} | R²={test_r2:.3f}")

# ---------------- Save Model ----------------
MODEL_PATH = MODEL_DIR / "lstm_multi_final.h5"
model.save(MODEL_PATH)
print(f"\n💾 Model saved → {MODEL_PATH}")

# ---------------- FUTURE FORECASTING ----------------
future_horizons = [1, 10, 30, 60]
future_results = {}

last_window = X[-1].copy()

for step in future_horizons:
    window = last_window.copy()
    preds = []

    for _ in range(step):
        p = model.predict(window.reshape(1, LOOKBACK, X.shape[2]), verbose=0)[0][0]
        preds.append(p)
        new_row = window[-1].copy()
        new_row[0] = p
        window = np.vstack([window[1:], new_row])

    future_results[step] = int(inverse(np.array(preds))[-1])

print("\n⏳ FUTURE FORECAST:")
for k, v in future_results.items():
    print(f"➡ {k} min: {v} vehicles/min")

# ---------------- Plot TRAIN ----------------
plt.figure(figsize=(14,5))
plt.plot(time_train, y_train_real, label="Actual")
plt.plot(time_train, train_pred, '--', label="Predicted")
plt.title(f"TRAIN — RMSE={train_rmse:.2f} | R²={train_r2:.3f}")
plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

# ---------------- Plot TEST + FUTURE ----------------
future_steps = 60
future_scaled = []
window = last_window.copy()

for _ in range(future_steps):
    p = model.predict(window.reshape(1, LOOKBACK, X.shape[2]), verbose=0)[0][0]
    future_scaled.append(p)
    new_row = window[-1].copy()
    new_row[0] = p
    window = np.vstack([window[1:], new_row])

future_real = inverse(np.array(future_scaled))
future_time = pd.date_range(df["timestamp"].iloc[-1], periods=future_steps+1, freq="1min")[1:]

plt.figure(figsize=(15,5))
plt.plot(time_test, y_test_real, label="Actual (Test)")
plt.plot(time_test, test_pred, '--', label="Predicted (Test)")
plt.plot(future_time, future_real, 'r--', label="Future Forecast")
plt.title(f"Future Traffic Forecast (next 60 min)\nR²={test_r2:.3f}, RMSE={test_rmse:.2f}")
plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

print("\n✅ Completed Successfully.")
