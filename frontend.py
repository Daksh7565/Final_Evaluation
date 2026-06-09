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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def _load_worker_model():
    """Load a dedicated model + tracker for each worker thread."""
    model = RFDETRBase()
    try:
        if torch.cuda.is_available():
            model = model.to("cuda")
    except Exception:
        pass
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
    return model, tracker


def _worker_process_route(shared_state, route_name, video_path, worker_id):
    """Thread worker: processes one route and pushes frames/stats to shared_state."""
    st_key = f"slot_{worker_id}"

    # Each worker gets its own DB session
    session = RouteSessionLocal()
    try:
        location = ROUTE_LOCATIONS.get(route_name, f"Camera {route_name}")
        route_obj = get_or_create_route(
            session, route_name,
            location=location,
            line_config={"type": "horizontal", "y_percent": 0.8}
        )

        # Check if already processed
        completed_run = session.query(RouteRun).filter(
            RouteRun.route_id == route_obj.id,
            RouteRun.end_time.isnot(None)
        ).first()

        if completed_run:
            with shared_state["lock"]:
                shared_state[st_key] = {
                    "status": "skipped",
                    "route_name": route_name,
                    "frame": None,
                    "frame_index": 0,
                    "total_frames": 0,
                    "counts": {},
                    "detections": 0,
                    "timestamp": "",
                    "fps": 0,
                    "elapsed": 0,
                    "message": f"{route_name} already processed"
                }
            return

        # Load dedicated model for this worker
        model, tracker = _load_worker_model()

        route_start = time.time()

        for result in process_route_generator(route_name, video_path, model, tracker, session, route_obj):
            with shared_state["lock"]:
                if result["status"] == "frame":
                    # Resize frame for display performance
                    frame = result["frame"]
                    h, w = frame.shape[:2]
                    max_w = 900
                    if w > max_w:
                        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))

                    elapsed = time.time() - route_start
                    fps = result["frame_index"] / elapsed if elapsed > 0 else 0

                    shared_state[st_key] = {
                        "status": "running",
                        "route_name": route_name,
                        "frame": frame,
                        "frame_index": result["frame_index"],
                        "total_frames": result["total_frames"],
                        "counts": result["counts"],
                        "detections": result["detections"],
                        "timestamp": result["timestamp"],
                        "fps": fps,
                        "elapsed": elapsed,
                        "message": f"Processing {route_name}..."
                    }

                elif result["status"] == "completed":
                    shared_state[st_key] = {
                        "status": "completed",
                        "route_name": route_name,
                        "frame": None,
                        "frame_index": result.get("frame_index", 0),
                        "total_frames": result.get("frame_index", 0),
                        "counts": result["counts"],
                        "detections": 0,
                        "timestamp": "",
                        "fps": 0,
                        "elapsed": time.time() - route_start,
                        "message": f"{route_name} completed — {result['counts']}"
                    }
                    shared_state["completed_count"] = shared_state.get("completed_count", 0) + 1

                elif result["status"] == "error":
                    shared_state[st_key] = {
                        "status": "error",
                        "route_name": route_name,
                        "frame": None,
                        "frame_index": 0,
                        "total_frames": 0,
                        "counts": {},
                        "detections": 0,
                        "timestamp": "",
                        "fps": 0,
                        "elapsed": 0,
                        "message": result["message"]
                    }
    finally:
        session.close()


def show_processing_page():
    """Run the AI model on videos 2-at-a-time with side-by-side live display."""
    st.title("🔬 Live Model Processing — Dual Stream")
    st.info("""
        The AI detection model processes **2 video routes simultaneously**. 
        Live frames with detection overlays are shown side-by-side below.
        Each slot runs independently with its own model instance.
    """)

    # Gather pending routes
    all_routes = list(ROUTE_VIDEOS.items())
    pending_routes = []
    for rname, vpath in all_routes:
        if not Path(vpath).exists():
            continue
        session = RouteSessionLocal()
        try:
            route_obj = session.query(Route).filter_by(name=rname).first()
            if route_obj:
                completed = session.query(RouteRun).filter(
                    RouteRun.route_id == route_obj.id,
                    RouteRun.end_time.isnot(None)
                ).first()
                if not completed:
                    pending_routes.append((rname, vpath))
            else:
                pending_routes.append((rname, vpath))
        finally:
            session.close()

    # If nothing pending, jump straight to done
    if not pending_routes:
        st.success("✅ All routes already processed!")
        nav_cols = st.columns([1, 1.5, 1])
        with nav_cols[1]:
            if st.button("🏠 Go to Dashboard", type="primary", use_container_width=True):
                load_route_names.clear()
                load_route_data.clear()
                st.session_state.page = "dashboard"
                st.rerun()
        return

    # Overall progress
    total_pending = len(pending_routes)
    st.markdown(f"**Routes pending:** {total_pending} | **Processing:** 2 at a time")
    overall_progress = st.progress(0, text="Preparing...")

    # ── Slot 1 Layout ──
    st.markdown("### 🎥 Active Processing Streams")
    slot_cols = st.columns(2)

    # Slot 0 placeholders
    with slot_cols[0]:
        slot0_header = st.empty()
        slot0_frame = st.empty()
        slot0_progress = st.empty()
        st.markdown("---")
        st.markdown("**📊 Slot 1 Stats**")
        slot0_frame_counter = st.empty()
        slot0_fps = st.empty()
        slot0_elapsed = st.empty()
        slot0_counts = st.empty()

    # Slot 1 placeholders
    with slot_cols[1]:
        slot1_header = st.empty()
        slot1_frame = st.empty()
        slot1_progress = st.empty()
        st.markdown("---")
        st.markdown("**📊 Slot 2 Stats**")
        slot1_frame_counter = st.empty()
        slot1_fps = st.empty()
        slot1_elapsed = st.empty()
        slot1_counts = st.empty()

    # Status log
    status_container = st.container()
    with status_container:
        st.markdown("### 📋 Processing Log")
        log_display = st.empty()

    log_messages = []

    def add_log(msg):
        log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(log_messages) > 20:
            log_messages.pop(0)
        log_display.markdown(
            "<div style='font-size: 0.82rem; color: #555; background: #f8f8f8; padding: 10px; border-radius: 6px; max-height: 200px; overflow-y: auto;'>" +
            "<br>".join(log_messages) +
            "</div>",
            unsafe_allow_html=True
        )

    # Shared state for inter-thread communication
    manager_state = {
        "lock": threading.Lock(),
        "slot_0": {"status": "idle", "route_name": "", "frame": None, "frame_index": 0,
                   "total_frames": 0, "counts": {}, "detections": 0, "timestamp": "",
                   "fps": 0, "elapsed": 0, "message": ""},
        "slot_1": {"status": "idle", "route_name": "", "frame": None, "frame_index": 0,
                   "total_frames": 0, "counts": {}, "detections": 0, "timestamp": "",
                   "fps": 0, "elapsed": 0, "message": ""},
        "completed_count": 0,
        "total_processed_count": total_pending - len(pending_routes)  # already done
    }

    # Process in pairs
    grand_start = time.time()
    total_routes_all = len(all_routes)
    routes_done = total_routes_all - total_pending  # pre-completed

    # Placeholders for slot headers
    slot0_header.markdown("🟡 **Slot 1:** Waiting...")
    slot1_header.markdown("🟡 **Slot 2:** Waiting...")

    # Break pending routes into pairs
    route_pairs = []
    for i in range(0, len(pending_routes), 2):
        pair = pending_routes[i:i+2]
        # Pad with None if odd number
        if len(pair) == 1:
            pair.append(None)
        route_pairs.append(pair)

    for pair in route_pairs:
        active_workers = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            for slot_id, route_item in enumerate(pair):
                if route_item is not None:
                    rname, vpath = route_item
                    add_log(f"▶️ Assigning {rname} to Slot {slot_id + 1}")
                    future = executor.submit(
                        _worker_process_route,
                        manager_state, rname, vpath, slot_id
                    )
                    futures[future] = (slot_id, rname)

            # Mark unused slot as done immediately
            for slot_id, route_item in enumerate(pair):
                if route_item is None:
                    with manager_state["lock"]:
                        manager_state[f"slot_{slot_id}"] = {
                            "status": "idle", "route_name": "", "frame": None,
                            "frame_index": 0, "total_frames": 0, "counts": {},
                            "detections": 0, "timestamp": "", "fps": 0,
                            "elapsed": 0, "message": "No route assigned"
                        }

            # Poll and display while workers are running
            active = True
            while active:
                active = False
                for future in futures:
                    if not future.done():
                        active = True
                        break

                # Read shared state and update UI
                with manager_state["lock"]:
                    s0 = manager_state["slot_0"].copy()
                    s1 = manager_state["slot_1"].copy()

                # ── Update Slot 0 ──
                if s0["route_name"]:
                    loc0 = ROUTE_LOCATIONS.get(s0["route_name"], "")
                    status_emoji_0 = {"running": "🔴", "completed": "✅", "skipped": "⏭️",
                                      "error": "❌", "idle": "🟡"}.get(s0["status"], "🟡")
                    slot0_header.markdown(f"{status_emoji_0} **Slot 1:** `{s0['route_name']}` — *{loc0}*")

                    if s0["frame"] is not None:
                        slot0_frame.image(s0["frame"], use_container_width=True)

                    if s0["total_frames"] > 0 and s0["status"] == "running":
                        prog = min(s0["frame_index"] / s0["total_frames"], 0.999)
                        slot0_progress.progress(prog, text=f"Frame {s0['frame_index']:,} / {s0['total_frames']:,}")
                    elif s0["status"] == "completed":
                        slot0_progress.progress(1.0, text="✅ Complete")
                    elif s0["status"] == "skipped":
                        slot0_progress.progress(1.0, text="⏭️ Skipped (already done)")

                    slot0_frame_counter.metric("Frame", f"{s0['frame_index']:,}")
                    slot0_fps.metric("Speed", f"{s0['fps']:.1f} FPS")
                    slot0_elapsed.metric("Elapsed", f"{s0['elapsed']:.0f}s")

                    counts_md = ""
                    for vclass, count in s0["counts"].items():
                        counts_md += f"- **{vclass.title()}:** {count}<br>"
                    slot0_counts.markdown(counts_md if counts_md else "*No vehicles detected yet*", unsafe_allow_html=True)

                # ── Update Slot 1 ──
                if s1["route_name"]:
                    loc1 = ROUTE_LOCATIONS.get(s1["route_name"], "")
                    status_emoji_1 = {"running": "🔴", "completed": "✅", "skipped": "⏭️",
                                      "error": "❌", "idle": "🟡"}.get(s1["status"], "🟡")
                    slot1_header.markdown(f"{status_emoji_1} **Slot 2:** `{s1['route_name']}` — *{loc1}*")

                    if s1["frame"] is not None:
                        slot1_frame.image(s1["frame"], use_container_width=True)

                    if s1["total_frames"] > 0 and s1["status"] == "running":
                        prog = min(s1["frame_index"] / s1["total_frames"], 0.999)
                        slot1_progress.progress(prog, text=f"Frame {s1['frame_index']:,} / {s1['total_frames']:,}")
                    elif s1["status"] == "completed":
                        slot1_progress.progress(1.0, text="✅ Complete")
                    elif s1["status"] == "skipped":
                        slot1_progress.progress(1.0, text="⏭️ Skipped (already done)")

                    slot1_frame_counter.metric("Frame", f"{s1['frame_index']:,}")
                    slot1_fps.metric("Speed", f"{s1['fps']:.1f} FPS")
                    slot1_elapsed.metric("Elapsed", f"{s1['elapsed']:.0f}s")

                    counts_md = ""
                    for vclass, count in s1["counts"].items():
                        counts_md += f"- **{vclass.title()}:** {count}<br>"
                    slot1_counts.markdown(counts_md if counts_md else "*No vehicles detected yet*", unsafe_allow_html=True)

                # Update overall progress
                with manager_state["lock"]:
                    newly_done = manager_state.get("completed_count", 0)
                routes_done = (total_routes_all - total_pending) + newly_done
                overall_progress.progress(
                    routes_done / total_routes_all,
                    text=f"Completed {routes_done} of {total_routes_all} routes"
                )

                # Brief sleep to let Streamlit breathe
                time.sleep(0.05)

            # All futures in this pair are done
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    slot_id, rname = futures[future]
                    add_log(f"❌ Exception in Slot {slot_id + 1} ({rname}): {e}")

        add_log(f"✅ Pair complete. Moving to next pair...")

    # All pairs complete
    grand_elapsed = time.time() - grand_start
    overall_progress.progress(1.0, text="All routes processed!")

    st.divider()
    st.balloons()
    st.success(f"🎉 All routes processed successfully in {grand_elapsed:.1f} seconds!")

    # Navigation to dashboard
    nav_cols = st.columns([1, 1.5, 1])
    with nav_cols[1]:
        if st.button("🏠 Go to Dashboard", type="primary", use_container_width=True):
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
