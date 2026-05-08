import streamlit as st
import pandas as pd
from sqlalchemy.orm import sessionmaker
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import cv2
import numpy as np
import torch
import time

from DB import engine, Route, RouteRun, get_time_series_counts, get_class_distribution
from agent import get_route_recommendation

# Import from routes.py: video sources, generator, and helpers
from routes import (
    ROUTE_VIDEOS,
    ROUTE_LOCATIONS,
    process_route_generator,
    get_or_create_route,
    SessionLocal as RouteSessionLocal,
    OUTPUT_VIDEO_DIR,
)
from rfdetr import RFDETRBase
from sort.Tracker import Sort
import yaml

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"error: the file not found ")

config = load_config()

# --- Page Configuration ---
st.set_page_config(
    page_title=config["dashboard"]["title"],
    page_icon=config["dashboard"]["icon"],
    layout="wide"
)

# --- Database Session ---
SessionLocal = sessionmaker(bind=engine)

# ============================================================================
# CACHED FUNCTIONS
# ============================================================================

@st.cache_data
def load_route_names():
    """Fetches a list of all route names from the database."""
    try:
        session = SessionLocal()
        routes = session.query(Route.name).all()
        return [route[0] for route in routes]
    finally:
        session.close()

@st.cache_data
def load_route_data(route_name):
    """Fetches all analytics data for a specific route."""
    try:
        session = SessionLocal()
        route = session.query(Route).filter_by(name=route_name).first()
        if not route:
            return None, None
        df_ts = get_time_series_counts(route.id)
        df_dist = get_class_distribution(route.id)
        if 'timestamp' in df_ts.columns:
            df_ts.rename(columns={'timestamp': 'minute'}, inplace=True)
        return df_ts, df_dist
    finally:
        session.close()

@st.cache_data
def get_video_thumbnail(video_path, max_width=400):
    """Extract a thumbnail (middle frame) from a video file."""
    path = Path(video_path)
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2 - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_rgb = cv2.resize(frame_rgb, (max_width, int(h * scale)))
    return frame_rgb

@st.cache_resource
def load_model_and_tracker():
    """Load the RFDETR model and SORT tracker once and cache them."""
    model = RFDETRBase()
    try:
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception:
        pass
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
    return model, tracker

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_route_processed(route_name):
    """Check if a route has a completed processing run in the DB."""
    try:
        session = SessionLocal()
        route = session.query(Route).filter_by(name=route_name).first()
        if not route:
            return False
        completed_run = session.query(RouteRun).filter(
            RouteRun.route_id == route.id,
            RouteRun.end_time.isnot(None)
        ).first()
        return completed_run is not None
    finally:
        session.close()

def get_busiest_hour(df_minute):
    """Calculates the busiest hour based on the minute with the highest peak traffic."""
    if df_minute is None or df_minute.empty:
        return "N/A"
    df_minute['minute'] = pd.to_datetime(df_minute['minute'])
    per_minute_total = df_minute.groupby(pd.Grouper(key='minute', freq='T'))['count'].sum().reset_index()
    if per_minute_total.empty:
        return "N/A"
    busiest_minute_row = per_minute_total.loc[per_minute_total['count'].idxmax()]
    busiest_timestamp = pd.to_datetime(busiest_minute_row['minute'])
    busiest_hour_val = busiest_timestamp.hour
    return f"{busiest_hour_val}:00 - {busiest_hour_val + 1}:00"

def get_most_common_vehicle(df_distribution):
    if df_distribution is None or df_distribution.empty:
        return "N/A"
    return df_distribution.loc[df_distribution['count'].idxmax()]['class_name'].title()

def resample_traffic_data(df, rule):
    """Resamples the traffic data to a specified time granularity."""
    if df is None or df.empty:
        return pd.DataFrame()
    df['minute'] = pd.to_datetime(df['minute'])
    df = df.set_index('minute')
    resampled_df = df.groupby('class_name').resample(rule)['count'].sum().reset_index()
    return resampled_df

# ============================================================================
# PAGE 1: VIDEO GALLERY
# ============================================================================

def show_gallery_page():
    """Display all raw videos in a grid with a 'Run the Model' button."""
    st.title("🎥 Traffic Video Analysis System")
    st.markdown("""
        <p style='font-size: 1.1rem; color: #888;'>
        Review all traffic camera feeds below. Click <b>"Run the Model"</b> to start 
        AI-powered vehicle detection and tracking across all routes.
        </p>
    """, unsafe_allow_html=True)

    # Check processing status for each route
    route_status = {rname: is_route_processed(rname) for rname in ROUTE_VIDEOS.keys()}
    all_processed = all(route_status.values())

    # --- Video Grid ---
    st.divider()
    st.subheader("📹 Raw Video Feeds")

    video_items = list(ROUTE_VIDEOS.items())
    cols_per_row = 3

    for row_idx in range(0, len(video_items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            item_idx = row_idx + col_idx
            if item_idx >= len(video_items):
                break
            rname, vpath = video_items[item_idx]
            location = ROUTE_LOCATIONS.get(rname, "Unknown Location")
            processed = route_status[rname]

            with col:
                with st.container(border=True):
                    # Thumbnail
                    thumb = get_video_thumbnail(vpath)
                    if thumb is not None:
                        st.image(thumb, use_container_width=True)
                    else:
                        st.warning("⚠️ Cannot load video preview")

                    # Route info
                    st.markdown(f"**{rname}**")
                    st.caption(f"📍 {location}")
                    st.caption(f"📁 `{Path(vpath).name}`")

                    # Status badge
                    if processed:
                        st.success("✅ Already Processed", icon="✅")
                    else:
                        st.warning("⏳ Pending", icon="⏳")

    # --- Bottom Action Bar ---
    st.divider()
    action_cols = st.columns([1, 1.5, 1])
    with action_cols[1]:
        if all_processed:
            st.info("All routes have already been processed. You can view the dashboard directly.")
            if st.button("🏠 Go to Dashboard", type="primary", use_container_width=True):
                st.session_state.page = "dashboard"
                st.rerun()
        else:
            pending_count = sum(1 for v in route_status.values() if not v)
            st.info(f"{pending_count} route(s) pending processing.")
            if st.button("🚀 Run the Model", type="primary", use_container_width=True):
                st.session_state.page = "processing"
                st.rerun()

# ============================================================================
# PAGE 2: LIVE PROCESSING
# ============================================================================

def show_processing_page():
    """Run the AI model on all videos with live frame display."""
    st.title("🔬 Live Model Processing")
    st.info("""
        The AI detection model (RFDETR + SORT tracker) is processing each video route sequentially. 
        Live frames with detection overlays are shown below. Please do not navigate away until processing completes.
    """)

    # Load model and tracker (cached)
    try:
        model, tracker = load_model_and_tracker()
    except Exception as e:
        st.error(f"Failed to load AI model: {e}")
        if st.button("← Back to Gallery"):
            st.session_state.page = "gallery"
            st.rerun()
        return

    # Overall progress
    total_routes = len(ROUTE_VIDEOS)
    overall_progress = st.progress(0, text="Starting...")

    # Layout: main video area + stats sidebar
    vid_col, stats_col = st.columns([3, 1])

    with vid_col:
        frame_placeholder = st.empty()
        route_header = st.empty()
        frame_progress = st.empty()

    with stats_col:
        st.markdown("### 📊 Live Stats")
        current_route_display = st.empty()
        frame_counter = st.empty()
        fps_display = st.empty()
        elapsed_display = st.empty()
        st.markdown("---")
        st.markdown("**Vehicle Counts**")
        vehicle_counts_display = st.empty()
        st.markdown("---")
        status_log = st.empty()

    # Processing log
    log_messages = []

    def add_log(msg):
        log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        # Keep last 6 messages
        if len(log_messages) > 6:
            log_messages.pop(0)
        status_log.markdown(
            "<div style='font-size: 0.8rem; color: #666; background: #f5f5f5; padding: 8px; border-radius: 4px;'>" +
            "<br>".join(log_messages) +
            "</div>",
            unsafe_allow_html=True
        )

    # Process each route
    completed_routes = 0
    grand_start = time.time()

    for idx, (rname, vpath) in enumerate(ROUTE_VIDEOS.items()):
        # Update overall progress
        progress_val = idx / total_routes
        overall_progress.progress(
            progress_val,
            text=f"Route {idx + 1} of {total_routes}: {rname}"
        )

        if not Path(vpath).exists():
            add_log(f"⚠️ Video not found: {vpath}")
            continue

        location = ROUTE_LOCATIONS.get(rname, f"Camera {rname}")
        route_header.markdown(f"### 🎥 {rname} — *{location}*")
        current_route_display.markdown(f"**Route:** `{rname}`")

        with RouteSessionLocal() as session:
            route_obj = get_or_create_route(
                session, rname,
                location=location,
                line_config={"type": "horizontal", "y_percent": 0.8}
            )

            # Check if already processed
            completed_run = session.query(RouteRun).filter(
                RouteRun.route_id == route_obj.id,
                RouteRun.end_time.isnot(None)
            ).first()

            if completed_run:
                add_log(f"⏭️ {rname} already processed — skipping")
                completed_routes += 1
                continue

            add_log(f"▶️ Starting {rname}...")
            route_start = time.time()
            total_yielded_frames = 0

            for result in process_route_generator(rname, vpath, model, tracker, session, route_obj):
                if result["status"] == "frame":
                    total_yielded_frames += 1
                    frame_idx = result["frame_index"]
                    total_f = result["total_frames"]
                    counts = result["counts"]

                    # Resize frame for display performance
                    display_frame = result["frame"]
                    h, w = display_frame.shape[:2]
                    max_w = 1280
                    if w > max_w:
                        display_frame = cv2.resize(display_frame, (max_w, int(h * max_w / w)))

                    frame_placeholder.image(display_frame, use_container_width=True)

                    # Update frame progress bar
                    if total_f > 0:
                        frame_progress.progress(
                            min(frame_idx / total_f, 0.999),
                            text=f"Frame {frame_idx:,} / {total_f:,}"
                        )
                    else:
                        frame_progress.progress(0, text=f"Frame {frame_idx:,}")

                    # Update stats
                    elapsed = time.time() - route_start
                    fps = frame_idx / elapsed if elapsed > 0 else 0

                    frame_counter.metric("Frame", f"{frame_idx:,}", f"of {total_f:,}" if total_f > 0 else "")
                    fps_display.metric("Speed", f"{fps:.1f} FPS")
                    elapsed_display.metric("Elapsed", f"{elapsed:.0f}s")

                    counts_md = ""
                    for vclass, count in counts.items():
                        counts_md += f"- **{vclass.title()}:** {count}<br>"
                    vehicle_counts_display.markdown(counts_md, unsafe_allow_html=True)

                elif result["status"] == "completed":
                    add_log(f"✅ {rname} completed — {result['counts']}")
                    completed_routes += 1

                elif result["status"] == "error":
                    add_log(f"❌ {rname} error: {result['message']}")

    # All routes complete
    overall_elapsed = time.time() - grand_start
    overall_progress.progress(1.0, text="All routes processed!")

    st.divider()
    st.balloons()
    st.success(f"🎉 All routes processed successfully in {overall_elapsed:.1f} seconds!")

    # Navigation to dashboard
    nav_cols = st.columns([1, 1.5, 1])
    with nav_cols[1]:
        if st.button("🏠 Go to Dashboard", type="primary", use_container_width=True):
            # Clear cached data so dashboard loads fresh
            load_route_names.clear()
            load_route_data.clear()
            st.session_state.page = "dashboard"
            st.rerun()

# ============================================================================
# PAGE 3: ANALYTICS DASHBOARD
# ============================================================================

def show_dashboard_page():
    """Display the full analytics dashboard (original frontend.py content)."""

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.title("🚦 Dashboard Controls")
        st.write("Select a route to analyze its traffic patterns.")

        route_names = load_route_names()
        if not route_names:
            st.error("No route data found. Run the model to process videos first.")
            if st.button("← Go to Video Gallery"):
                st.session_state.page = "gallery"
                st.rerun()
            st.stop()

        selected_route = st.selectbox(
            "Select a Route",
            options=route_names,
            label_visibility="collapsed"
        )

        st.info("This dashboard provides real-time analytics from processed video feeds.")

        st.markdown("---")
        if st.button("← Back to Video Gallery", use_container_width=True):
            st.session_state.page = "gallery"
            st.rerun()

    # --- Main Application ---
    st.title("AI Traffic Analysis Dashboard")
    st.subheader("Real-time monitoring and intelligent route recommendations")

    # --- AI Agent Recommendation Section ---
    with st.container(border=True):
        st.header("🤖 AI Agent Recommendation")

        if st.button("Recommend the Best Route Now", type="primary", use_container_width=True):
            with st.spinner("The AI agent is analyzing the latest traffic data from all routes..."):
                recommendation_text = get_route_recommendation()
                st.success("**Analysis Complete! Here is the recommendation:**")
                st.markdown(recommendation_text)
        else:
            st.info("Click the button to get a real-time route recommendation from the AI agent.")

    # --- Route-Specific Analytics Section ---
    st.divider()

    if selected_route:
        st.header(f"📈 Analytics Dashboard for: `{selected_route}`")

        df_ts, df_distribution = load_route_data(selected_route)

        if df_ts is not None and not df_ts.empty:
            # --- Key Metrics (KPIs) ---
            total_vehicles = int(df_distribution['count'].sum())
            vehicle_types = len(df_distribution)
            busiest_hour = get_busiest_hour(df_ts.copy())
            common_vehicle = get_most_common_vehicle(df_distribution)

            kpi_cols = st.columns(4)
            kpi_cols[0].metric(label="Total Vehicles Counted", value=total_vehicles)
            kpi_cols[1].metric(label="Vehicle Types Detected", value=vehicle_types)
            kpi_cols[2].metric(label="Busiest Hour", value=busiest_hour)
            kpi_cols[3].metric(label="Most Common Vehicle", value=common_vehicle)

            st.divider()

            # --- Live Snapshot Section ---
            st.subheader("📸 Live Snapshot")
            snapshot_col, control_col = st.columns([0.8, 0.2])
            with snapshot_col:
                snapshot_path = Path(f"processed_frames/{selected_route}_latest.jpg")
                if snapshot_path.exists():
                    st.image(str(snapshot_path), caption=f"Last updated snapshot for {selected_route}")
                else:
                    st.info("Live snapshot will appear here once video processing for this route begins.")
            with control_col:
                st.button("Refresh Snapshot", use_container_width=True)

            # --- Charts in Tabs ---
            tab1, tab2 = st.tabs(["📊 Traffic Flow Over Time", "🚗 Vehicle Distribution"])

            with tab1:
                st.subheader("Traffic Flow — with 5-Period Rolling Average")

                time_granularity = st.select_slider(
                    "Select Time Granularity",
                    options=["10 Seconds", "30 Seconds", "1 Minute", "5 Minutes", "1 Hour"],
                    value="1 Minute"
                )
                time_mapping = {
                    "10 Seconds": "10S", "30 Seconds": "30S", "1 Minute": "T",
                    "5 Minutes": "5T", "1 Hour": "H"
                }
                resample_rule = time_mapping[time_granularity]
                df_resampled = resample_traffic_data(df_ts.copy(), resample_rule)

                if not df_resampled.empty:
                    time_format = '%H:%M'
                    if resample_rule in ["10S", "30S"]:
                        time_format = '%H:%M:%S'
                    elif resample_rule == "H":
                        time_format = '%H:00'

                    df_resampled['time_label'] = df_resampled['minute'].dt.strftime(time_format)
                    pivot_df = df_resampled.pivot_table(
                        index='time_label', columns='class_name',
                        values='count', fill_value=0
                    )
                    pivot_df['total'] = pivot_df.sum(axis=1)
                    pivot_df['rolling_5period'] = pivot_df['total'].rolling(window=5, min_periods=1).mean()

                    fig = go.Figure()
                    for vclass in df_resampled['class_name'].unique():
                        if vclass in pivot_df.columns:
                            fig.add_trace(go.Scatter(
                                x=pivot_df.index, y=pivot_df[vclass],
                                mode='lines+markers', name=vclass, opacity=0.6
                            ))

                    fig.add_trace(go.Scatter(
                        x=pivot_df.index, y=pivot_df['rolling_5period'],
                        mode='lines', name="5-Period Rolling Avg", line=dict(width=4)
                    ))

                    if not pivot_df.empty:
                        peak_time = pivot_df['total'].idxmax()
                        peak_value = pivot_df['total'].max()
                        fig.add_annotation(
                            x=peak_time, y=peak_value,
                            text=f"Peak: {int(peak_value)} vehicles",
                            showarrow=True, arrowhead=2
                        )

                    fig.update_layout(
                        height=450,
                        xaxis_title=f"Time (Granularity: {time_granularity})",
                        yaxis_title="Vehicle Count",
                        hovermode="x unified",
                        template="plotly_dark",
                        legend_title="Vehicle Types"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No time-series data available for the selected route.")

            with tab2:
                st.subheader("Vehicle Class Distribution")
                if not df_distribution.empty:
                    col1, col2 = st.columns([0.6, 0.4])
                    with col1:
                        st.write("Count by Vehicle Type")
                        bar_chart_data = df_distribution.set_index('class_name')
                        st.bar_chart(bar_chart_data)
                    with col2:
                        st.write("Proportion of Vehicle Types")
                        fig_pie = px.pie(
                            df_distribution, values='count',
                            names='class_name', hole=.3
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("No vehicle distribution data available.")

            # --- Raw Data ---
            with st.expander("View Raw Data Tables"):
                st.write(f"**Per-Second Counts for `{selected_route}` (Raw Data):**")
                st.dataframe(df_ts)
                st.write(f"**Class Distribution for `{selected_route}`:**")
                st.dataframe(df_distribution)
        else:
            st.warning(f"No vehicle data has been recorded for route: `{selected_route}`. "
                       "Please check if the video processing was successful.")

# ============================================================================
# MAIN ROUTER
# ============================================================================

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "gallery"

# Route to the appropriate page
page = st.session_state.page

if page == "gallery":
    show_gallery_page()
elif page == "processing":
    show_processing_page()
elif page == "dashboard":
    show_dashboard_page()
