"""
Final Dataset Cleaner (Working Fix)
 - Keeps real weather
 - Fixes zeros and missing traffic values
 - Smooths noise
 - Keeps rain column
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

BASE_DIR = Path(__file__).resolve().parent.parent

# Detect weather file
REAL = BASE_DIR / "data" / "agg" / "traffic_weather_real.csv"
SYNTH = BASE_DIR / "data" / "agg" / "traffic_weather_merged.csv"

if REAL.exists():
    INPUT = REAL
    print("📌 Using REAL weather dataset.")
elif SYNTH.exists():
    INPUT = SYNTH
    print("📌 Using SYNTHETIC weather dataset.")
else:
    raise FileNotFoundError("❌ No weather dataset found.")

OUTPUT = BASE_DIR / "data" / "agg" / "traffic_cleaned.csv"

print(f"📥 Loading dataset → {INPUT}")
df = pd.read_csv(INPUT)

# Ensure timestamp is datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Set timestamp as index for time interpolation
df = df.set_index("timestamp")

# ---- Normalize rain column ----
if "rain_1h" in df.columns:
    df.rename(columns={"rain_1h": "rain_mm"}, inplace=True)
elif "precipitation" in df.columns:
    df.rename(columns={"precipitation": "rain_mm"}, inplace=True)
else:
    print("⚠ No rain column found — creating one.")
    df["rain_mm"] = 0.0

# ---- Replace unrealistic zeros with NaN ----
df.loc[df["total_vehicles"] == 0, "total_vehicles"] = np.nan

print("⏳ Interpolating missing values...")
df["total_vehicles"] = df["total_vehicles"].interpolate(method="time")

# ---- Smooth noise ----
print("📉 Applying smoothing filter...")
df["total_vehicles"] = savgol_filter(df["total_vehicles"], window_length=21, polyorder=3)

# Remove negatives after smoothing
df["total_vehicles"] = df["total_vehicles"].clip(lower=0)

# Reset index back
df = df.reset_index()

# Save cleaned file
df.to_csv(OUTPUT, index=False)
print(f"✅ CLEANED dataset saved → {OUTPUT}")
print(f"📏 Rows: {len(df)} | Columns: {list(df.columns)}")
