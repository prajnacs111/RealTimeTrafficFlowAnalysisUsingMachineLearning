# weather_fetcher_real.py
import requests
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
TRAFFIC_FILE = BASE_DIR / "data" / "agg" / "traffic_per_min.csv"
OUT_FILE = BASE_DIR / "data" / "agg" / "traffic_weather_real.csv"

# Replace with the coordinates of your traffic location
LAT = 47.6162     # Bellevue, WA
LON = -122.1903

# Weather variables we want
VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation"
]

COLUMN_MAP = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "precipitation": "rain"
}

# ---------------- FETCH & MERGE ----------------
def fetch_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(VARS),
        "timezone": "auto"
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])

def main():
    df = pd.read_csv(TRAFFIC_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    start = str(df["timestamp"].dt.date.min())
    end = str(df["timestamp"].dt.date.max())

    print(f"📡 Fetching real weather: {start} → {end} [{LAT}, {LON}]")
    
    weather = fetch_weather(start, end)

    weather["timestamp"] = pd.to_datetime(weather["time"])
    weather = weather.drop(columns=["time"])
    
    # Rename columns
    weather.rename(columns=COLUMN_MAP, inplace=True)

    # Resample and interpolate per minute
    weather = weather.set_index("timestamp").resample("1min").interpolate().reset_index()

    # Merge with traffic data
    merged = df.merge(weather, on="timestamp", how="left")

    # Final cleaning
    merged["rain"] = merged["rain"].fillna(0)
    merged["wind_speed"] = merged["wind_speed"].fillna(method="ffill").fillna(0)
    merged["humidity"] = merged["humidity"].fillna(method="ffill")
    merged["temp"] = merged["temp"].fillna(method="ffill")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)

    print(f"✅ Weather merged successfully → {OUT_FILE}")
    print(f"🔢 Total rows: {len(merged)}")

if __name__ == "__main__":
    main()
