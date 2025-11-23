# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import json
from datetime import datetime, timezone
from pathlib import Path
import yaml

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ==============================================================================
# 2. CONFIGURATION & INITIALIZATION
# ==============================================================================

# --- a. Directory and Path Setup ---
def load_config(config_path="config.yaml"):
    try:
        with open(config_path,'r',encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
config=load_config()

LOG_DIR = Path(config["paths"]["log_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = Path(config["paths"]["database"])
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- b. Database Setup ---
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
engine = create_engine(
    DATABASE_URL, echo=False, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ==============================================================================
# 3. DATABASE MODELS
# ==============================================================================


class Route(Base):
    __tablename__ = "routes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    location = Column(String, nullable=True)
    line_config = Column(JSON, nullable=True)  # e.g. {"type":"horizontal","y":800}
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    runs = relationship("RouteRun", back_populates="route")


class RouteRun(Base):
    __tablename__ = "route_runs"
    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    raw_video_path = Column(Text, nullable=True)

    route = relationship("Route", back_populates="runs")
    vehicle_tracks = relationship("VehicleTrack", back_populates="run")
    counts = relationship("CountEvent", back_populates="run")


class VehicleTrack(Base):
    __tablename__ = "vehicle_tracks"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("route_runs.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=True)
    last_seen = Column(DateTime(timezone=True), nullable=True)
    track_info = Column(JSON, nullable=True)  # e.g. {"bboxes":[...], "speeds": [...]}

    run = relationship("RouteRun", back_populates="vehicle_tracks")


class CountEvent(Base):
    __tablename__ = "counts"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("route_runs.id"), nullable=False)
    route_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    class_name = Column(String, nullable=False)
    count_increment = Column(Integer, nullable=False)

    run = relationship("RouteRun", back_populates="counts")


# ==============================================================================
# 4. LOGGING FUNCTIONS
# ==============================================================================


def log_frame_json(
    route_id: str, frame_index: int, detections: list, counts_incremented: dict
):
    """Logs detection data for a single frame to a JSON file.

    Args:
        route_id (str): The identifier for the route.
        frame_index (int): The index of the video frame.
        detections (list): A list of detection dictionaries.
        counts_incremented (dict): A dictionary of class counts that were incremented.
    """
    output_data = {
        "route_id": route_id,
        "frame_index": frame_index,
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "detections": detections,
        "counts_incremented": counts_incremented,
    }
    log_path = LOG_DIR / f"{route_id}.json"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(output_data) + "\n")


# ==============================================================================
# 5. DATA AGGREGATION & ANALYSIS
# ==============================================================================


def get_per_minute_counts(route_id, run_id=None):
    """Retrieves per-minute vehicle counts for a given route."""
    session = SessionLocal()
    query = (
        session.query(
            func.strftime("%Y-%m-%d %H:%M", CountEvent.timestamp).label("minute"),
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count"),
        )
        .filter(CountEvent.route_id == route_id)
        .group_by("minute", CountEvent.class_name)
        .order_by("minute")
    )

    if run_id:
        query = query.filter(CountEvent.run_id == run_id)

    df = pd.DataFrame(query.all(), columns=["minute", "class_name", "count"])
    session.close()
    return df


def get_hourly_counts(route_id):
    """Retrieves hourly vehicle counts for a given route."""
    session = SessionLocal()
    query = (
        session.query(
            func.strftime("%Y-%m-%d %H", CountEvent.timestamp).label("hour"),
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count"),
        )
        .filter(CountEvent.route_id == route_id)
        .group_by("hour", CountEvent.class_name)
        .order_by("hour")
    )
    df = pd.DataFrame(query.all(), columns=["hour", "class_name", "count"])
    session.close()
    return df



def get_time_series_counts(route_id, run_id=None):
    """
    Retrieves per-second vehicle counts for a given route.
    This provides the highest granularity for flexible frontend resampling.
    """
    session = SessionLocal()
    query = (
        session.query(
            # MODIFICATION: Group by the second to get high-resolution data
            func.strftime("%Y-%m-%d %H:%M:%S", CountEvent.timestamp).label("timestamp"),
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count"),
        )
        .filter(CountEvent.route_id == route_id)
        # MODIFICATION: Group by the full timestamp string
        .group_by("timestamp", CountEvent.class_name)
        .order_by("timestamp")
    )

    if run_id:
        query = query.filter(CountEvent.run_id == run_id)

    df = pd.DataFrame(query.all(), columns=["timestamp", "class_name", "count"])
    session.close()
    return df


def get_per_30_second_counts(route_id, run_id=None):
    """Retrieves per-30-second vehicle counts for a given route."""
    session = SessionLocal()
    query = (
        session.query(
            (func.strftime('%Y-%m-%d %H:%M:', CountEvent.timestamp) or
             (func.strftime('%S', CountEvent.timestamp) / 30 * 30)).label("time_bucket"),
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count"),
        )
        .filter(CountEvent.route_id == route_id)
        .group_by("time_bucket", CountEvent.class_name)
        .order_by("time_bucket")
    )
    if run_id:
        query = query.filter(CountEvent.run_id == run_id)
    df = pd.DataFrame(query.all(), columns=["time_bucket", "class_name", "count"])
    session.close()
    return df


def get_per_10_second_counts(route_id, run_id=None):
    """Retrieves per-10-second vehicle counts for a given route."""
    session = SessionLocal()
    query = (
        session.query((func.strftime('%Y-%m-%d %H:%M:', CountEvent.timestamp) or (func.strftime('%S', CountEvent.timestamp) / 10 * 10)).label("time_bucket"),
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count"),
        )
        .filter(CountEvent.route_id == route_id)
        .group_by("time_bucket", CountEvent.class_name)
        .order_by("time_bucket")
    )
    if run_id:
        query = query.filter(CountEvent.run_id == run_id)
    df = pd.DataFrame(query.all(), columns=["time_bucket", "class_name", "count"])
    session.close()
    return df
def get_class_distribution(route_id):
    """Retrieves the distribution of vehicle classes for a given route."""
    session = SessionLocal()
    query = (
        session.query(
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count"),
        )
        .filter(CountEvent.route_id == route_id)
        .group_by(CountEvent.class_name)
    )
    df = pd.DataFrame(query.all(), columns=["class_name", "count"])
    session.close()
    return df


def get_rolling_5min(df_per_minute):
    """Calculates a 5-minute rolling average from per-minute count data."""
    if df_per_minute.empty:
        return df_per_minute

    df = df_per_minute.copy()
    df["minute"] = pd.to_datetime(df["minute"])

    pivot_df = df.pivot_table(
        index="minute", columns="class_name", values="count", fill_value=0
    )

    rolling_avg = pivot_df.rolling("5min").mean()
    rolling_avg.reset_index(inplace=True)
    return rolling_avg


# ==============================================================================
# 6. DATA EXPORT
# ==============================================================================


def export_to_csv(df, filename):
    """Exports a DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Saved CSV → {filename}")