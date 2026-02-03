"""
signal_logic.py
----------------
This module decides which lane should get the green signal next
based on real-time factors:
- current vehicle count
- forecast traffic (LSTM output)
- anomaly detection (accident / abnormal flow)
- emergency detection (sirens + YOLO emergency)
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LaneState:
    """Represents the state of each lane as processed in controller."""
    lane_id: int
    name: str
    current: int          # Current detected vehicle count
    forecast: int         # LSTM predicted vehicle count
    emergency: bool       # Siren + Emergency vehicle detection
    anomaly: bool         # Autoencoder spike abnormal condition


def decide_next_phase(
    lanes: List[LaneState],
    current_lane_id: int,
    min_green: int,
    max_green: int
) -> Tuple[int, int, dict]:
    """
    Decide next green lane using priority rules.

    Returns:
        next_lane_id: int
        duration: int (seconds)
        metadata: dict -> for the dashboard {"reason": "..."}
    """

    # ------------------------------
    # PRIORITY 1: Emergency Vehicle
    # ------------------------------
    emergency_lanes = [lane for lane in lanes if lane.emergency]

    if emergency_lanes:
        lane = max(emergency_lanes, key=lambda x: x.current + x.forecast)
        return lane.lane_id, max_green, {"reason": f"🚨 Emergency detected on {lane.name}"}

    # ------------------------------
    # PRIORITY 2: Anomaly Detection
    # (accidents, stalled vehicles)
    # ------------------------------
    anomaly_lanes = [lane for lane in lanes if lane.anomaly]

    if anomaly_lanes:
        lane = max(anomaly_lanes, key=lambda x: x.current + x.forecast)
        return lane.lane_id, max_green - 5, {"reason": f"⚠ Anomaly detected on {lane.name}"}

    # ------------------------------
    # PRIORITY 3: Forecasted Traffic
    # (LSTM prediction ahead)
    # ------------------------------
    sorted_by_forecast = sorted(lanes, key=lambda x: x.forecast, reverse=True)

    highest = sorted_by_forecast[0]

    # If highest predicted lane is not current → switch
    if highest.lane_id != current_lane_id:
        duration = min_green + min(int(highest.forecast / 5), max_green - min_green)
        return highest.lane_id, duration, {"reason": f"📈 Traffic forecast: {highest.name}"}

    # ------------------------------
    # PRIORITY 4: Normal Load (Current Vehicles)
    # ------------------------------
    sorted_by_current = sorted(lanes, key=lambda x: x.current, reverse=True)
    top_lane = sorted_by_current[0]

    if top_lane.lane_id != current_lane_id:
        duration = min_green + min(int(top_lane.current / 5), max_green - min_green)
        return top_lane.lane_id, duration, {"reason": f"🚦 Higher present demand: {top_lane.name}"}

    # ------------------------------
    # DEFAULT CASE:
    # Stay on current lane (no reason to switch)
    # ------------------------------
    duration = min_green
    return current_lane_id, duration, {"reason": "⏳ Stable traffic — holding current lane"}
