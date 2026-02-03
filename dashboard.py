# dashboard.py
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st

# ======================================================
# PATHS / GLOBALS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
STATUS_FILE = SCRIPTS_DIR / "controller_status.json"
LOGS_ROOT = SCRIPTS_DIR / "logs"
LIVE_FEED_DIR = SCRIPTS_DIR / "live_feed"

LIVE_FEED_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="Real Time Traffic Flow Analysis and Prediction System",
    page_icon="🚦",
    layout="wide",
)

# ======================================================
# SESSION STATE
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "lanes" not in st.session_state:
    # default config for 4 lanes
    st.session_state.lanes = {
        i: {
            "enabled": (i == 1),  # only lane 1 enabled by default
            "mode": "local",      # local / upload / camera
            "source": f"data/videos/test_real/{i}.mp4",
        }
        for i in range(1, 5)
    }

if "yolo_started" not in st.session_state:
    st.session_state.yolo_started = False

if "controller_started" not in st.session_state:
    st.session_state.controller_started = False

if "auto_refresh_live" not in st.session_state:
    st.session_state.auto_refresh_live = True


# ======================================================
# SIMPLE AUTH
# ======================================================
VALID_USERS = {
    "admin": "admin123",
    "traffic": "traffic2025",
}


def login_box() -> bool:
    """Simple login form on Home page."""
    if st.session_state.logged_in:
        st.success("✅ Logged in as admin.")
        return True

    st.subheader("🔐 Admin Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in VALID_USERS and VALID_USERS[user] == pwd:
            st.session_state.logged_in = True
            st.success("Login successful. Use sidebar to navigate.")
        else:
            st.error("Invalid credentials")

    return st.session_state.logged_in


# ======================================================
# BACKEND HELPERS
# ======================================================
def render_lane_config(lane_id: int):
    """Lane configuration UI (source selection)."""
    cfg = st.session_state.lanes[lane_id]

    with st.expander(f"Lane {lane_id} Configuration", expanded=(lane_id == 1)):
        cfg["enabled"] = st.checkbox(
            f"Enable Lane {lane_id}",
            value=cfg["enabled"],
            key=f"lane{lane_id}_enabled",
        )

        if not cfg["enabled"]:
            st.info("Lane disabled.")
            return

        st.write("Select input type")
        mode_label_map = {
            "local": "🔴 Local Video File",
            "upload": "⤴ Upload Video",
            "camera": "📹 Camera Feed",
        }

        current_index = list(mode_label_map.keys()).index(cfg["mode"])
        mode_label = st.radio(
            "",
            list(mode_label_map.values()),
            index=current_index,
            key=f"lane{lane_id}_mode",
        )
        cfg["mode"] = next(k for k, v in mode_label_map.items() if v == mode_label)

        # --- Mode-specific source handling ---
        if cfg["mode"] == "local":
            cfg["source"] = st.text_input(
                "Local video path",
                value=cfg["source"],
                key=f"lane{lane_id}_src",
            )

        elif cfg["mode"] == "upload":
            uploaded = st.file_uploader(
                "Upload video",
                type=["mp4", "avi", "mov", "mkv"],
                key=f"u{lane_id}",
            )
            if uploaded is not None:
                upload_dir = BASE_DIR / "data" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_path = upload_dir / f"lane{lane_id}_{uploaded.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded.read())
                cfg["source"] = str(save_path)
                st.success(f"Saved to {save_path}")

        elif cfg["mode"] == "camera":
            cam_id = st.number_input(
                "Camera Index",
                min_value=0,
                max_value=10,
                value=int(cfg["source"]) if cfg["source"].isdigit() else 0,
                key=f"cam{lane_id}",
            )
            cfg["source"] = str(cam_id)


def start_yolo(enable_audio: bool = True):
    """Start YOLO lane processes for all enabled lanes."""
    for lane_id, cfg in st.session_state.lanes.items():
        if not cfg["enabled"] or not cfg["source"]:
            continue

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "test1_yolo_multi.py"),
            "--lane", str(lane_id),
            "--source", cfg["source"],
        ]
        if enable_audio:
            cmd.append("--audio")

        # fire-and-forget
        subprocess.Popen(
            cmd,
            cwd=str(SCRIPTS_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    st.session_state.yolo_started = True
    st.success("YOLO processes started for enabled lanes.")


def start_controller():
    """Start smart_traffic_controller.py as background process."""
    subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / "smart_traffic_controller.py")],
        cwd=str(SCRIPTS_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state.controller_started = True
    st.success("Smart Traffic Controller started.")


def load_status() -> Dict[str, Any]:
    """Read controller_status.json safely."""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # being written; skip this cycle
            return {}
    return {}


def latest_lane_csv(lane_id: int) -> Optional[Path]:
    """Get latest YOLO log CSV for analytics."""
    lane_dir = LOGS_ROOT / f"lane_{lane_id}"
    if not lane_dir.exists():
        return None
    files = list(lane_dir.glob("*.csv"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


# ======================================================
# RENDER SECTIONS
# ======================================================
def page_home():
    """Page 1 – Title + Login."""
    st.title(
        "🚦 Real Time Traffic Flow Analysis and Prediction System\n"
        "using Dynamic Traffic Signal Control and Anomaly Detection"
    )

    st.markdown("---")
    st.write(
        """
This dashboard is the **control center** for your smart traffic system:

- YOLOv8 vehicle + emergency detection per lane  
- Siren detection from audio  
- LSTM-based traffic flow prediction (10, 30, 60 minutes)  
- Hybrid anomaly detection (accident / sudden spike)  
- Weather-aware adaptive signal control  
"""
    )

    st.markdown("### 🔐 Admin Access")
    login_box()


def page_live_monitor():
    """Page 2 – Live monitoring, weather, previews, dynamic stats."""
    st.header("📡 Live Monitoring & Signal Control")

    if not st.session_state.logged_in:
        st.warning("Please login on the Home page to use this section.")
        return

    st.markdown("### 1️⃣ Lane Input Configuration")
    cols = st.columns(2)
    with cols[0]:
        for i in (1, 3):  # lane 1 & 3 in left column
            render_lane_config(i)
    with cols[1]:
        for i in (2, 4):  # lane 2 & 4 in right column
            render_lane_config(i)

    st.markdown("---")
    st.markdown("### 2️⃣ System Control")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("▶ Start YOLO (all enabled lanes)"):
            start_yolo(enable_audio=True)

    with c2:
        if st.button("▶ Start Smart Traffic Controller"):
            start_controller()

    with c3:
        st.session_state.auto_refresh_live = st.checkbox(
            "Auto-refresh live page", value=st.session_state.auto_refresh_live
        )

    st.markdown("---")
    st.markdown("### 3️⃣ Live Conditions")

    status = load_status()
    if not status:
        st.info("Waiting for controller_status.json … Start the controller first.")
        return

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Active Lane", f"Lane {status['active_lane']}")
    colB.metric("Time Remaining", f"{status['remaining_time']} sec")
    colC.metric("Reason", status["reason"])
    weather = status.get("weather", {})
    colD.metric(
        "Weather",
        f"{weather.get('temp', 0):.1f}°C, wind {weather.get('wind_speed', 0):.1f} m/s",
    )

    # --- Lane summary table ---
    rows = []
    for lane in status.get("lanes", []):
        rows.append({
            "Lane": lane["lane"],
            "Current": lane["count_now"],
            "Now": lane["forecast_10"],  # we don't have 1-step in JSON; using 10m as proxy
            "+10m": lane["forecast_10"],
            "+30m": lane["forecast_30"],
            "+60m": lane["forecast_60"],
            "Status": (
                "🚨 Emergency" if lane["emergency"]
                else "⚠ Anomaly" if lane["anomaly"]
                else "Normal"
            ),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.markdown("### 4️⃣ YOLO Preview (Hybrid Mode)")

    # 2×2 grid for preview images
    preview_cols = st.columns(2)
    for idx, lane_id in enumerate(sorted(st.session_state.lanes.keys())):
        cfg = st.session_state.lanes[lane_id]
        with preview_cols[idx % 2]:
            st.subheader(f"📍 Lane {lane_id}")
            if not cfg["enabled"]:
                st.info("Lane disabled.")
                continue

            img_path = LIVE_FEED_DIR / f"lane_{lane_id}.jpg"
            if img_path.exists():
                st.image(str(img_path), caption=f"YOLO preview – Lane {lane_id}")
            else:
                st.info("Waiting for YOLO preview frame…")

    if st.session_state.auto_refresh_live:
        st.info("🔄 Auto-refreshing live data and previews…")
        time.sleep(1)
        st.rerun()


def page_analytics():
    """Page 3 – Historical graphs + forecast view."""
    st.header("📊 Traffic Analytics & Forecasts")

    if not st.session_state.logged_in:
        st.warning("Please login on the Home page to use this section.")
        return

    lane_ids = sorted(st.session_state.lanes.keys())
    lane_id = st.selectbox("Select Lane", lane_ids)

    csv_path = latest_lane_csv(lane_id)
    if not csv_path:
        st.info(f"No log data yet for Lane {lane_id}. Run YOLO first.")
        return

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    st.subheader(f"📈 Lane {lane_id} – Vehicle Count Over Time")
    st.line_chart(
        df.set_index("timestamp")["total_vehicle_count"],
        use_container_width=True,
    )

    # Forecast snapshot from controller_status.json
    status = load_status()
    lane_info = None
    for l in status.get("lanes", []):
        if int(l["lane"]) == int(lane_id):
            lane_info = l
            break

    if lane_info:
        st.subheader("🔮 Current Forecast Snapshot (LSTM)")
        forecast_df = pd.DataFrame({
            "Horizon": ["+10 min", "+30 min", "+60 min"],
            "Vehicles": [
                lane_info["forecast_10"],
                lane_info["forecast_30"],
                lane_info["forecast_60"],
            ],
        })
        st.bar_chart(
            forecast_df.set_index("Horizon"),
            use_container_width=True,
        )
    else:
        st.info("No forecast snapshot yet from controller for this lane.")


# ======================================================
# MAIN NAVIGATION
# ======================================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home / Login", "Live Monitor", "Analytics"],
)

if page == "Home / Login":
    page_home()
elif page == "Live Monitor":
    page_live_monitor()
elif page == "Analytics":
    page_analytics()
