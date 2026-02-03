# smart_traffic_controller.py

import time
import json
import requests
from pathlib import Path
from glob import glob
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import warnings

# --- Silence noisy warnings (FutureWarning + sklearn feature names) ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X has feature names")

# ======================================================
# PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_ROOT = BASE_DIR / "scripts" / "logs"
HIST_DATA_FILE = BASE_DIR / "data" / "agg" / "traffic_cleaned.csv"
CONTROLLER_STATUS = BASE_DIR / "scripts" / "controller_status.json"

# ======================================================
# MODEL CONFIG
# ======================================================
LSTM_MODEL_PATH = MODELS_DIR / "lstm_multi_final.h5"
LSTM_SCALER_PATH = MODELS_DIR / "lstm_scaler.pkl"
LSTM_LOOKBACK = 30
FEATURES = ["total_vehicles", "temp", "humidity", "wind_speed", "rain_mm"]

AE_MODEL_PATH = MODELS_DIR / "anomaly_autoencoder.keras"
AE_SCALER_PATH = MODELS_DIR / "anomaly_scaler.pkl"
AE_THRESHOLD = float(np.load(MODELS_DIR / "anomaly_threshold.npy"))
SPIKE_THRESHOLD = float(np.load(MODELS_DIR / "anomaly_spike_threshold.npy"))
AE_WINDOW = 20

# ======================================================
# TIMING LOGIC
# ======================================================
REFRESH_SECONDS = 1          # controller loop step
MIN_GREEN = 8               # lower bound
MAX_GREEN = 120             # safety upper bound
MAX_WAIT = 120              # fairness rule (sec)

# ======================================================
# WEATHER API (live + fallback)
# ======================================================
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=13.20875&longitude=75.0228&current_weather=true"
)


@dataclass
class LaneState:
    lane_id: int
    count_now: int
    forecast_now: float
    p10: float
    p30: float
    p60: float
    emergency: bool
    anomaly: bool


# ======================================================
# LOAD MODELS
# ======================================================
print("\n📥 Loading AI models...")

lstm = load_model(LSTM_MODEL_PATH, compile=False)
lstm_scaler = joblib.load(LSTM_SCALER_PATH)

ae = load_model(AE_MODEL_PATH, compile=False)
ae_scaler = joblib.load(AE_SCALER_PATH)

print("✅ Models loaded.\n")

# Historical fallback for weather
hist = pd.read_csv(HIST_DATA_FILE)
hist["timestamp"] = pd.to_datetime(hist["timestamp"])
hist = hist.sort_values("timestamp").reset_index(drop=True)


# ======================================================
# HELPERS
# ======================================================
def get_weather() -> dict:
    """Fetch live weather; fallback to most recent historical weather."""
    try:
        resp = requests.get(WEATHER_URL, timeout=2)
        data = resp.json()
        return {
            "temp": float(data["current_weather"]["temperature"]),
            "humidity": 50.0,  # API free tier doesn't give humidity → assume mid
            "wind_speed": float(data["current_weather"]["windspeed"]),
            "rain_mm": 0.0,
        }
    except Exception:
        tail = hist.tail(1)
        return {
            "temp": float(tail["temp"]),
            "humidity": float(tail["humidity"]),
            "wind_speed": float(tail["wind_speed"]),
            "rain_mm": float(tail["rain_mm"]),
        }


def discover_lanes() -> List[int]:
    """Find lane_* folders dynamically."""
    lanes = sorted({int(p.name.split("_")[1]) for p in LOGS_ROOT.glob("lane_*")})
    return lanes


def latest_csv(lane: int) -> Optional[str]:
    """Get latest CSV for a lane (from YOLO logger)."""
    files = glob(str(LOGS_ROOT / f"lane_{lane}" / "*.csv"))
    if not files:
        return None
    return max(files, key=lambda f: Path(f).stat().st_mtime)


def load_lane(path: str, rows: int = 300) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.tail(rows).reset_index(drop=True)


# ======================================================
# LSTM + Anomaly
# ======================================================
def lstm_forecast(series: pd.Series, weather: dict):
    """Predict current + 10/30/60 min using trained LSTM model."""
    s = series.tail(LSTM_LOOKBACK).astype(float)

    if len(s) == 0:
        # If lane is totally empty, just use historical mean
        mean_val = float(hist["total_vehicles"].mean())
        s = pd.Series([mean_val] * LSTM_LOOKBACK)
    elif len(s) < LSTM_LOOKBACK:
        pad = [s.iloc[0]] * (LSTM_LOOKBACK - len(s))
        s = pd.Series(pad + list(s))

    block = pd.DataFrame({
        "total_vehicles": s.values,
        "temp": [weather["temp"]] * LSTM_LOOKBACK,
        "humidity": [weather["humidity"]] * LSTM_LOOKBACK,
        "wind_speed": [weather["wind_speed"]] * LSTM_LOOKBACK,
        "rain_mm": [weather["rain_mm"]] * LSTM_LOOKBACK,
    })

    def inverse(x: float) -> float:
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = x
        return float(lstm_scaler.inverse_transform(dummy)[0, 0])

    # Now prediction
    scaled = lstm_scaler.transform(block.values).reshape(1, LSTM_LOOKBACK, -1)
    now_scaled = lstm.predict(scaled, verbose=0)[0][0]
    now = max(0.0, inverse(now_scaled))

    # Autoregressive roll-out
    def roll(steps: int) -> float:
        w = block.copy()
        last = now
        for _ in range(steps):
            x = lstm_scaler.transform(w.values).reshape(1, LSTM_LOOKBACK, -1)
            p_scaled = lstm.predict(x, verbose=0)[0][0]
            last = max(0.0, inverse(p_scaled))
            new = w.iloc[-1].copy()
            new["total_vehicles"] = last
            w = pd.concat([w.iloc[1:], new.to_frame().T], ignore_index=True)
        return last

    return now, roll(10), roll(30), roll(60)


def detect_anomaly(counts: pd.Series) -> bool:
    """Hybrid anomaly detection: autoencoder error + spike check."""
    if len(counts) < AE_WINDOW:
        return False

    r = counts.tail(AE_WINDOW).astype(float)
    smooth = r.rolling(10, min_periods=1).mean()

    weather_block = hist.tail(AE_WINDOW)
    block = pd.DataFrame({
        "traffic_smooth": smooth.values,
        "temp": weather_block["temp"].values,
        "humidity": weather_block["humidity"].values,
        "wind_speed": weather_block["wind_speed"].values,
        "rain_mm": weather_block["rain_mm"].values,
    })

    scaled = ae_scaler.transform(block.values)
    recon = ae.predict(scaled.reshape(1, -1), verbose=0)
    err = float(np.mean((scaled.flatten() - recon.flatten()) ** 2))

    spike = float(abs(r.iloc[-1] - smooth.iloc[-1]))

    return (err > AE_THRESHOLD) or (spike > SPIKE_THRESHOLD)


# ======================================================
# CONTROLLER LOGIC
# ======================================================
def pick_lane(lanes: List[LaneState], current: int, wait: dict):
    """
    Decide next green lane and duration.
    Priority:
      1) Emergency
      2) Anomaly
      3) Fairness (no lane starves)
      4) Highest load (current + forecast)
    """
    if not lanes:
        # No data; keep current lane minimally
        return current, MIN_GREEN, "Waiting for lane data"

    emergencies = [l for l in lanes if l.emergency]
    if emergencies:
        chosen = max(emergencies, key=lambda x: x.count_now)
        base = 20
        duration = int(np.clip(base, MIN_GREEN, MAX_GREEN))
        wait[chosen.lane_id] = 0
        return chosen.lane_id, duration, "🚨 Emergency Priority"

    anomalies = [l for l in lanes if l.anomaly]
    if anomalies:
        chosen = max(anomalies, key=lambda x: x.count_now)
        base = 15
        duration = int(np.clip(base, MIN_GREEN, MAX_GREEN))
        wait[chosen.lane_id] = 0
        return chosen.lane_id, duration, "⚠ Accident / Jam"

    overdue = [l.lane_id for l in lanes if wait.get(l.lane_id, 0) >= MAX_WAIT]
    if overdue:
        chosen_id = overdue[0]
        base = 15
        duration = int(np.clip(base, MIN_GREEN, MAX_GREEN))
        wait[chosen_id] = 0
        return chosen_id, duration, "🔄 Fairness Rule: Prevent starvation"

    best = max(lanes, key=lambda x: x.count_now + x.forecast_now)

    if best.lane_id == current:
        base = 10
        duration = int(np.clip(base, MIN_GREEN, MAX_GREEN))
        return current, duration, "↪ Hold (still highest load)"

    base = 12
    duration = int(np.clip(base, MIN_GREEN, MAX_GREEN))
    wait[current] = wait.get(current, 0) + duration
    wait[best.lane_id] = 0
    return best.lane_id, duration, "🔀 Switching to heavier lane"


# ======================================================
# MAIN LOOP
# ======================================================
def main():
    lanes = discover_lanes()
    if not lanes:
        print("❌ ERROR: No YOLO lane logs found. Run test1_yolo_multi.py first.")
        return

    print(f"🚦 Controller Running — Lanes: {lanes}")

    current = lanes[0]
    remaining = 10
    reason = "Initialization"
    wait_time = {l: 0 for l in lanes}

    while True:
        weather = get_weather()
        states: List[LaneState] = []

        # ---- Gather state per lane ----
        for lane in lanes:
            path = latest_csv(lane)
            if not path:
                continue

            df = load_lane(path)
            if "total_vehicle_count" not in df.columns:
                continue

            counts = df["total_vehicle_count"]

            emergency = False
            if "emergency_vehicle" in df.columns:
                emergency = emergency or (df["emergency_vehicle"].tail(5) > 0).any()
            if "audio_siren_detected" in df.columns:
                emergency = emergency or bool(df["audio_siren_detected"].tail(5).any())

            anomaly = detect_anomaly(counts)
            now, p10, p30, p60 = lstm_forecast(counts, weather)

            states.append(
                LaneState(
                    lane_id=lane,
                    count_now=int(counts.tail(5).mean()),
                    forecast_now=now,
                    p10=p10,
                    p30=p30,
                    p60=p60,
                    emergency=emergency,
                    anomaly=anomaly,
                )
            )

        # ---- Decide lane only when phase ends ----
        if remaining <= 0:
            current, duration, reason = pick_lane(states, current, wait_time)
            remaining = duration
            print(
                f"\n🟢 {datetime.now().strftime('%H:%M:%S')} → "
                f"Lane {current} for {duration}s | {reason}"
            )

        # ---- Update waiting times ----
        for lane_id in wait_time:
            if lane_id != current:
                wait_time[lane_id] += REFRESH_SECONDS

        # --------- SAFE JSON ATOMIC WRITE ---------
        temp_file = CONTROLLER_STATUS.with_suffix(".tmp")

        payload = {
            "timestamp": datetime.now().isoformat(),
            "active_lane": int(current),
            "remaining_time": int(max(remaining, 0)),
            "reason": reason,
            "weather": {
                "temp": float(weather["temp"]),
                "humidity": float(weather["humidity"]),
                "wind_speed": float(weather["wind_speed"]),
                "rain_mm": float(weather["rain_mm"]),
            },
            "lanes": [
                {
                    "lane": int(l.lane_id),
                    "count_now": int(l.count_now),
                    "forecast_10": int(l.p10),
                    "forecast_30": int(l.p30),
                    "forecast_60": int(l.p60),
                    "emergency": bool(l.emergency),
                    "anomaly": bool(l.anomaly),
                }
                for l in states
            ],
        }

        with open(temp_file, "w") as f:
            json.dump(payload, f, indent=2)

        temp_file.replace(CONTROLLER_STATUS)
        # ------------------------------------------

        remaining -= REFRESH_SECONDS
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Controller stopped.")
