import os
import json
import time
import cv2
import streamlit as st

import database
import ui
from detector import VehicleDetector
from speed import SpeedEstimator

st.set_page_config(page_title="Traffic AI", page_icon="🚦", layout="wide")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
SNAP_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def init_state():
    database.init_db()
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "seen_violation_ids" not in st.session_state:
        st.session_state.seen_violation_ids = set()


def get_video_source(config):
    if config["input_mode"] == "Webcam":
        return 0
    uploaded = config.get("uploaded_file")
    if uploaded is None:
        return None
    path = os.path.join(SNAP_DIR, "_input_" + uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def run_detection(config):
    source = get_video_source(config)
    if source is None:
        st.warning("Please upload a video or choose webcam.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Could not open video source. Check the file or webcam availability.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    try:
        detector = VehicleDetector(
            model_path=config.get("model", "yolo11n.pt"),
            confidence=config["confidence"],
        )
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return

    estimator = SpeedEstimator(
        pixel_distance=config["pixel_distance"],
        real_distance_m=config["real_distance"],
        fps=fps,
    )

    limits = config.get("vehicle_limits", {})
    default_limit = config["speed_limit"]

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    stop_button = st.button("Stop Detection", key="stop_btn")

    frame_idx = 0
    frame_skip = config.get("frame_skip", 1)

    while cap.isOpened() and st.session_state.running:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % max(1, frame_skip) != 0:
            continue

        detections, inference_ms = detector.track(frame)

        for det in detections:
            tid = det["track_id"]
            cx, cy = det["center"]
            speed = estimator.update(tid, cx, cy, frame_idx)
            vehicle_type = det["class_name"]
            x1, y1, x2, y2 = det["box"]
            limit = limits.get(vehicle_type, default_limit)
            speeding = speed > limit and speed > 0

            color = (0, 0, 255) if speeding else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid} | {vehicle_type} | {speed:.0f} km/h"
            if speeding:
                label += " | SPEEDING"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            database.upsert_vehicle(tid, vehicle_type, speed)

            if speeding and not database.has_violation(tid):
                snap_path = os.path.join(SNAP_DIR, f"violation_{tid}_{int(time.time())}.jpg")
                cv2.imwrite(snap_path, frame)
                database.insert_violation(tid, vehicle_type, speed, limit, snap_path)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        device = detector.device.upper()
        stats_placeholder.caption(f"FPS: {fps:.0f} | Inference: {inference_ms:.1f} ms | Device: {device}")

        if stop_button:
            st.session_state.running = False
            break

    cap.release()


def page_dashboard():
    st.title("Dashboard")
    stats = database.get_stats()
    ui.render_metrics(stats)
    st.metric("Current Speed Limit", f"{st.session_state.config.get('speed_limit', 50)} km/h")
    ui.render_dashboard_charts(stats)


def page_detection(config):
    st.title("Detection")
    col1, col2 = st.columns(2)
    if col1.button("Start Detection"):
        st.session_state.running = True
    if col2.button("Reset Session Data"):
        database.clear_session()
        st.success("Session data cleared.")

    if st.session_state.running:
        run_detection(config)


def page_violations():
    st.title("Violations")
    ui.render_violations_table()


def page_ai_assistant():
    st.title("AI Assistant")
    st.caption("Ask questions about the collected traffic data. Answers are grounded in real SQLite results.")
    ui.render_chat()


def page_settings(config):
    st.title("Settings")
    config["model"] = st.selectbox("Model", ["yolo11n.pt", "yolo11s.pt"], index=0)
    config["frame_skip"] = st.number_input("Frame skip", min_value=1, max_value=10, value=config.get("frame_skip", 1))
    if st.button("Save Settings"):
        save_config(config)
        st.success("Settings saved to config.json")


def main():
    init_state()
    config = ui.render_sidebar(st.session_state.config)
    st.session_state.config = config

    page = st.sidebar.radio("Page", ["Dashboard", "Detection", "Violations", "AI Assistant", "Settings"])

    if page == "Dashboard":
        page_dashboard()
    elif page == "Detection":
        page_detection(config)
    elif page == "Violations":
        page_violations()
    elif page == "AI Assistant":
        page_ai_assistant()
    elif page == "Settings":
        page_settings(config)


if __name__ == "__main__":
    main()
