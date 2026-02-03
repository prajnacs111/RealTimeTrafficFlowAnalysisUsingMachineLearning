import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st

# ==============================
# PATHS & CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"
STREAM_DIR = BASE_DIR / "data" / "streams"
STREAM_DIR.mkdir(parents=True, exist_ok=True)

CONTROLLER_STATUS_FILE = SCRIPTS_DIR / "controller_status.json"
YOLO_SCRIPT = SCRIPTS_DIR / "test1_yolo_multi.py"
CONTROLLER_SCRIPT = SCRIPTS_DIR / "smart_traffic_controller.py"

REFRESH_SECONDS = 5
MIN_LANES = 2
MAX_LANES = 4

st.set_page_config(page_title="Real Time Smart Traffic Control", layout="wide")

PROJECT_TITLE = (
    "Real Time Traffic Flow Analysis and Prediction System "
    "using Dynamic Traffic Signal Control and Anomaly Detection"
)

# ==============================
# SESSION STATE
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Login"
if "lane_sources" not in st.session_state:
    st.session_state.lane_sources = {lane: None for lane in range(1, MAX_LANES + 1)}
if "controller_started" not in st.session_state:
    st.session_state.controller_started = False
if "yolo_started" not in st.session_state:
    st.session_state.yolo_started = {}


# ==============================
# HELPERS
# ==============================
def read_controller_state():
    if CONTROLLER_STATUS_FILE.exists():
        try:
            return json.loads(CONTROLLER_STATUS_FILE.read_text())
        except Exception:
            return None
    return None


def start_yolo_for_lane(lane_id: int, source: str, use_audio: bool = True):
    if not YOLO_SCRIPT.exists():
        st.error(f"YOLO script not found: {YOLO_SCRIPT}")
        return

    cmd = [
        sys.executable,
        str(YOLO_SCRIPT),
        "--lane",
        str(lane_id),
        "--source",
        str(source),
    ]
    if use_audio:
        cmd.append("--audio")

    try:
        subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        st.session_state.yolo_started[lane_id] = True
    except Exception as e:
        st.error(f"Failed to start YOLO for Lane {lane_id}: {e}")


def start_controller():
    if st.session_state.controller_started:
        return
    if not CONTROLLER_SCRIPT.exists():
        st.error(f"Controller script not found: {CONTROLLER_SCRIPT}")
        return

    try:
        subprocess.Popen(
            [sys.executable, str(CONTROLLER_SCRIPT)],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        st.session_state.controller_started = True
    except Exception as e:
        st.error(f"Failed to start controller: {e}")


def latest_lane_csv(lane_id: int):
    pattern = str(LOGS_DIR / f"lane{lane_id}_*.csv")
    files = glob(pattern)
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.title("🛣️ Smart Traffic")
    if st.session_state.logged_in:
        st.session_state.current_page = st.radio(
            "Navigate",
            ["Login", "Live Control", "Analytics & Logout"],
            index=["Login", "Live Control", "Analytics & Logout"].index(
                st.session_state.current_page
            ),
        )
    else:
        st.session_state.current_page = "Login"
        st.info("Please login to access control and analytics.")


# ==============================
# PAGE: LOGIN
# ==============================
def login_page():
    st.title("🔐 Admin Login")
    st.markdown(f"### {PROJECT_TITLE}")
    st.markdown("---")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.current_page = "Live Control"
            st.success("Login successful ✅")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid username or password ❌")


# ==============================
# PAGE: LIVE CONTROL
# ==============================
def live_control_page():
    st.title("🚦 Live Smart Traffic Control Center")
    st.markdown(f"### {PROJECT_TITLE}")
    st.markdown("---")

    st.subheader("1️⃣ Lane Input Configuration")

    enabled_lanes = []

    for lane_id in range(1, MAX_LANES + 1):
        with st.expander(
            f"Lane {lane_id} Configuration", expanded=(lane_id <= MIN_LANES)
        ):
            enabled = st.checkbox(
                f"Enable Lane {lane_id}",
                value=(lane_id <= MIN_LANES),
                key=f"enable_lane_{lane_id}",
            )

            if not enabled:
                st.session_state.lane_sources[lane_id] = None
                continue

            enabled_lanes.append(lane_id)

            mode = st.radio(
                "Select input type",
                ["📁 Local Video File", "⬆ Upload Video", "📷 Camera Feed"],
                key=f"mode_lane_{lane_id}",
            )

            source_value = None

            if mode == "📁 Local Video File":
                source_value = st.text_input(
                    "Video path",
                    value=str(
                        BASE_DIR
                        / "data"
                        / "videos"
                        / "test_real"
                        / f"{lane_id}.mp4"
                    ),
                    key=f"path_lane_{lane_id}",
                )

            elif mode == "⬆ Upload Video":
                uploaded = st.file_uploader(
                    "Upload video (max 200 MB)",
                    type=["mp4", "avi", "mov"],
                    key=f"upload_lane_{lane_id}",
                )
                if uploaded is not None:
                    if uploaded.size > 200 * 1024 * 1024:
                        st.error("File too large. Limit 200 MB.")
                    else:
                        upload_dir = BASE_DIR / "data" / "uploads"
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        out_path = upload_dir / f"lane{lane_id}_{uploaded.name}"
                        with open(out_path, "wb") as f:
                            f.write(uploaded.read())
                        source_value = str(out_path)
                        st.success(f"Uploaded to {out_path}")

            elif mode == "📷 Camera Feed":
                cam_type = st.selectbox(
                    "Camera type",
                    ["Laptop Webcam (0)", "USB Camera (1)", "RTSP / IP URL"],
                    key=f"cam_type_lane_{lane_id}",
                )
                if cam_type == "Laptop Webcam (0)":
                    source_value = "0"
                elif cam_type == "USB Camera (1)":
                    source_value = "1"
                else:
                    source_value = st.text_input(
                        "RTSP / IP URL", key=f"rtsp_lane_{lane_id}"
                    )

            st.session_state.lane_sources[lane_id] = {
                "mode": mode,
                "source": source_value,
            }

    st.markdown("---")

    st.subheader("2️⃣ System Control")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start YOLO"):
            if not enabled_lanes:
                st.warning("Enable at least one lane.")
            else:
                for lane_id in enabled_lanes:
                    cfg = st.session_state.lane_sources[lane_id]
                    if cfg and cfg["source"]:
                        start_yolo_for_lane(lane_id, cfg["source"], use_audio=True)
                st.success("YOLO started (check separate windows).")

    with col2:
        if st.button("▶ Start Smart Traffic Controller"):
            start_controller()
            if st.session_state.controller_started:
                st.success("Controller started.")

    st.markdown("---")

    st.subheader("3️⃣ Live Conditions")

    controller_state = read_controller_state()
    if not controller_state:
        st.info("Waiting for controller data...")
        time.sleep(2)
        st.rerun()

    colA, colB, colC = st.columns(3)
    colA.metric("Active Lane", f"Lane {controller_state.get('active_lane', 'N/A')}")
    colB.metric(
        "Next Update", f"{controller_state.get('refresh', REFRESH_SECONDS)} sec"
    )
    colC.metric("Reason", controller_state.get("reason", "N/A"))

    rows = []
    for lane in controller_state["lanes"]:
        status = (
            "🚨 Emergency"
            if lane["emergency"]
            else ("⚠ Anomaly" if lane["anomaly"] else "🟢 Normal")
        )
        rows.append(
            [
                lane["id"],
                lane["name"],
                lane["current"],
                lane["forecast_now"],
                lane["forecast_10"],
                lane["forecast_30"],
                lane["forecast_60"],
                status,
            ]
        )

    st.dataframe(
        pd.DataFrame(
            rows,
            columns=[
                "Lane",
                "Name",
                "Current",
                "Now",
                "+10m",
                "+30m",
                "+60m",
                "Status",
            ],
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("4️⃣ YOLO Preview (Hybrid Mode)")

    cols = st.columns(2)
    for idx, lane_id in enumerate(range(1, 5)):
        with cols[idx % 2]:
            st.markdown(f"**📍 Lane {lane_id}**")
            img_path = STREAM_DIR / f"lane{lane_id}.jpg"
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
            else:
                st.info("Waiting for YOLO preview frame...")

    st.info("🔄 Auto-refreshing to update live data and previews...")
    time.sleep(REFRESH_SECONDS)
    st.rerun()


# ==============================
# PAGE: ANALYTICS & LOGOUT
# ==============================
def analytics_page():
    st.title("📈 Analytics & History")
    st.markdown(f"### {PROJECT_TITLE}")
    st.markdown("---")

    lane_id = st.selectbox("Select lane", [1, 2, 3, 4])

    csv_path = latest_lane_csv(lane_id)
    if not csv_path:
        st.warning("No YOLO logs yet for this lane.")
    else:
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns and "total_vehicle_count" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            st.line_chart(
                df.set_index("timestamp")["total_vehicle_count"], height=300
            )
        st.dataframe(df.tail(25), use_container_width=True)

    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_page = "Login"
        st.session_state.controller_started = False
        st.session_state.yolo_started = {}
        st.success("Logged out.")
        time.sleep(1)
        st.rerun()


# ==============================
# ROUTER
# ==============================
if st.session_state.current_page == "Login":
    login_page()
else:
    if not st.session_state.logged_in:
        st.warning("Please login first.")
        login_page()
    elif st.session_state.current_page == "Live Control":
        live_control_page()
    elif st.session_state.current_page == "Analytics & Logout":
        analytics_page()
