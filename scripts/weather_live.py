# weather_live.py
"""
Live weather fetcher using Open-Meteo (no API key needed)
Returns data in the SAME feature format used by LSTM:
    temp, humidity, wind_speed, rain_mm
"""

import requests
from datetime import datetime, timezone
from pathlib import Path

# You can adjust this later from config or UI
LAT = 47.6162    # example: Bellevue, WA  (replace with your junction)
LON = -122.1903

def fetch_live_weather(lat: float = LAT, lon: float = LON) -> dict:
    """
    Fetch latest hourly weather from Open-Meteo and return a dict:
    {
        "temp": float (°C),
        "humidity": float (%),
        "wind_speed": float (km/h),
        "rain_mm": float (mm),
        "timestamp": ISO string
    }
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "auto"
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    hums = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    rains = hourly.get("precipitation", [])

    if not times:
        raise RuntimeError("No hourly weather returned from Open-Meteo.")

    # Take the *last* available hour as “current”
    idx = len(times) - 1

    return {
        "timestamp": times[idx],
        "temp": float(temps[idx]),
        "humidity": float(hums[idx]),
        "wind_speed": float(winds[idx]),
        "rain_mm": float(rains[idx]),
    }


if __name__ == "__main__":
    print("📡 Fetching live weather from Open-Meteo...")
    w = fetch_live_weather()
    print("✅ Live weather:")
    for k, v in w.items():
        print(f"  {k}: {v}")
