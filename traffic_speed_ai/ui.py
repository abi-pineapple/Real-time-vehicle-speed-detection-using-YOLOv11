import streamlit as st
import plotly.express as px
import pandas as pd
import database


def render_sidebar(config):
    st.sidebar.title("🚦 TRAFFIC AI")
    st.sidebar.markdown("---")

    config["speed_limit"] = st.sidebar.number_input(
        "Speed Limit (km/h)", min_value=10, max_value=200, value=config.get("speed_limit", 50)
    )
    config["confidence"] = st.sidebar.slider(
        "Confidence", 0.1, 0.9, config.get("confidence", 0.4), 0.05
    )

    with st.sidebar.expander("Vehicle-specific limits"):
        limits = config.get("vehicle_limits", {"Car": 50, "Motorcycle": 50, "Bus": 40, "Truck": 40})
        for v in limits:
            limits[v] = st.number_input(v, min_value=10, max_value=200, value=limits[v], key=f"lim_{v}")
        config["vehicle_limits"] = limits

    st.sidebar.markdown("---")
    config["input_mode"] = st.sidebar.radio("Input", ["Video", "Webcam"])
    if config["input_mode"] == "Video":
        config["uploaded_file"] = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    with st.sidebar.expander("Calibration"):
        config["pixel_distance"] = st.number_input(
            "Pixel distance (px)", min_value=1, value=config.get("pixel_distance", 400)
        )
        config["real_distance"] = st.number_input(
            "Real distance (m)", min_value=0.1, value=float(config.get("real_distance", 20.0))
        )
    st.sidebar.caption(
        "⚠️ Speed accuracy depends on camera angle, placement, FPS, video "
        "quality and calibration accuracy."
    )
    return config


def render_metrics(stats):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Vehicles", stats["total_vehicles"])
    c2.metric("Speeding", stats["speeding_vehicles"])
    c3.metric("Avg Speed", f"{stats['average_speed']} km/h")
    c4.metric("Max Speed", f"{stats['max_speed']} km/h")


def render_dashboard_charts(stats):
    col1, col2 = st.columns(2)
    with col1:
        if stats["vehicles_by_type"]:
            df = pd.DataFrame(
                {"Type": list(stats["vehicles_by_type"].keys()), "Count": list(stats["vehicles_by_type"].values())}
            )
            st.plotly_chart(px.pie(df, names="Type", values="Count", title="Vehicle Types"), use_container_width=True)
        else:
            st.info("No vehicles detected yet.")
    with col2:
        vehicles = database.get_all_vehicles()
        if vehicles:
            df = pd.DataFrame(vehicles)
            st.plotly_chart(
                px.histogram(df, x="max_speed", nbins=15, title="Speed Distribution"),
                use_container_width=True,
            )
        else:
            st.info("No speed data yet.")

    if stats["violations_by_type"]:
        df = pd.DataFrame(
            {"Type": list(stats["violations_by_type"].keys()), "Violations": list(stats["violations_by_type"].values())}
        )
        st.plotly_chart(px.bar(df, x="Type", y="Violations", title="Violations by Type"), use_container_width=True)


def render_violations_table():
    violations = database.get_all_violations()
    if not violations:
        st.info("No violations recorded yet.")
        return
    df = pd.DataFrame(violations)
    search = st.text_input("Filter by vehicle type")
    if search:
        df = df[df["vehicle_type"].str.contains(search, case=False)]
    st.dataframe(df, use_container_width=True)

    snap_options = [v["snapshot"] for v in violations if v.get("snapshot")]
    if snap_options:
        chosen = st.selectbox("View snapshot", snap_options)
        if chosen:
            st.image(chosen, caption=chosen)


def render_chat():
    import ai
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ask about the traffic data...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        answer = ai.ask(question)
        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)
