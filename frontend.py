import streamlit as st
import pandas as pd
from sqlalchemy.orm import sessionmaker
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from DB import engine, Route, get_time_series_counts, get_class_distribution
from agent import get_route_recommendation
import yaml        

def load_config(config_path="config.yaml"):
    try:
        with open(config_path,'r',encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"error: the file not found ")
config=load_config()

# --- Page Configuration ---
st.set_page_config(
    page_title=config["dashboard"]["title"],
    page_icon=config["dashboard"]["icon"],
    layout="wide"
)

# --- Database Session ---
SessionLocal = sessionmaker(bind=engine)

# --- Caching Data Loading Functions ---
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
        
        # Use the high-resolution, per-second data function
        df_ts = get_time_series_counts(route.id) 
        df_dist = get_class_distribution(route.id)
        
        if 'timestamp' in df_ts.columns:
            df_ts.rename(columns={'timestamp': 'minute'}, inplace=True)
            
        return df_ts, df_dist
    finally:
        session.close()

# --- Helper Functions for Metrics & Data Processing ---
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
    
    return f"{busiest_hour_val}:00 - {busiest_hour_val+1}:00"

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

# --- Sidebar for Controls ---
with st.sidebar:
    st.title("🚦 Dashboard Controls")
    st.write("Select a route to analyze its traffic patterns.")
    
    route_names = load_route_names()
    if not route_names:
        st.error("No route data found. Run `routes.py` to process videos.")
        st.stop()
        
    selected_route = st.selectbox(
        "Select a Route",
        options=route_names,
        label_visibility="collapsed"
    )
    
    st.info("This dashboard provides real-time analytics from processed video feeds.")

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
                if resample_rule in ["10S", "30S"]: time_format = '%H:%M:%S'
                elif resample_rule == "H": time_format = '%H:00'
                
                df_resampled['time_label'] = df_resampled['minute'].dt.strftime(time_format)
                pivot_df = df_resampled.pivot_table(index='time_label', columns='class_name', values='count', fill_value=0)
                pivot_df['total'] = pivot_df.sum(axis=1)
                pivot_df['rolling_5period'] = pivot_df['total'].rolling(window=5, min_periods=1).mean()

                fig = go.Figure()
                for vclass in df_resampled['class_name'].unique():
                    if vclass in pivot_df.columns:
                        fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[vclass], mode='lines+markers', name=vclass, opacity=0.6))
                
                fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df['rolling_5period'], mode='lines', name="5-Period Rolling Avg", line=dict(width=4)))

                if not pivot_df.empty:
                    peak_time = pivot_df['total'].idxmax()
                    peak_value = pivot_df['total'].max()
                    fig.add_annotation(x=peak_time, y=peak_value, text=f"Peak: {int(peak_value)} vehicles", showarrow=True, arrowhead=2)

                fig.update_layout(height=450, xaxis_title=f"Time (Granularity: {time_granularity})", yaxis_title="Vehicle Count", hovermode="x unified", template="plotly_dark", legend_title="Vehicle Types")
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
                    fig_pie = px.pie(df_distribution, values='count', names='class_name', hole=.3)
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
        st.warning(f"No vehicle data has been recorded for route: `{selected_route}`. Please check if the video processing was successful.")