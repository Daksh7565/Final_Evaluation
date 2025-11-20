# db.py
from sqlalchemy import create_engine,Column,Integer,String,DateTime,JSON,ForeignKey,Text,func
from sqlalchemy.orm import sessionmaker, scoped_session, relationship,declarative_base
from pathlib import Path
import json
from pathlib import Path
from datetime import datetime, timezone
from datetime import datetime, timedelta

import pandas as pd

from sqlalchemy.sql import func



def log_frame_json(route_id: str, frame_index: int, detections: list, counts_incremented: dict):
    """
    detections: list of dicts:
      {"track_id": int, "class": str, "bbox":[x1,y1,x2,y2], "centroid":[cx,cy], "crossed_line":bool}
    """
    out = {
        "route_id": route_id,
        "frame_index": frame_index,
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "detections": detections,
        "counts_incremented": counts_incremented
    }
    path = LOG_DIR / f"{route_id}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")





# -------------------------------------------------------
# 1. Per-minute aggregates
# -------------------------------------------------------
def get_per_minute_counts(route_id, run_id=None):
    session = SessionLocal()

    q = (
        session.query(
            func.strftime("%Y-%m-%d %H:%M", CountEvent.timestamp).label("minute"),
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count")
        )
        .filter(CountEvent.route_id == route_id)
    )

    if run_id:
        q = q.filter(CountEvent.run_id == run_id)

    q = q.group_by("minute", CountEvent.class_name).order_by("minute")

    df = pd.DataFrame(q.all(), columns=["minute", "class_name", "count"])
    session.close()
    return df


# -------------------------------------------------------
# 2. Hourly aggregates
# -------------------------------------------------------
def get_hourly_counts(route_id):
    session = SessionLocal()

    q = (
        session.query(
            func.strftime("%Y-%m-%d %H", CountEvent.timestamp).label("hour"),
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count")
        )
        .filter(CountEvent.route_id == route_id)
        .group_by("hour", CountEvent.class_name)
        .order_by("hour")
    )

    df = pd.DataFrame(q.all(), columns=["hour", "class_name", "count"])
    session.close()
    return df


# -------------------------------------------------------
# 3. Class-wise distribution
# -------------------------------------------------------
def get_class_distribution(route_id):
    session = SessionLocal()

    q = (
        session.query(
            CountEvent.class_name,
            func.sum(CountEvent.count_increment).label("count")
        )
        .filter(CountEvent.route_id == route_id)
        .group_by(CountEvent.class_name)
    )

    df = pd.DataFrame(q.all(), columns=["class_name", "count"])
    session.close()
    return df


# -------------------------------------------------------
# 4. 5-minute rolling averages
# -------------------------------------------------------
def get_rolling_5min(df_per_minute):
    if df_per_minute.empty:
        return df_per_minute

    df = df_per_minute.copy()
    df["minute"] = pd.to_datetime(df["minute"])

    # pivot into: minute | car | truck | bus
    pivot = df.pivot_table(index="minute", columns="class_name", values="count", fill_value=0)

    rolling = pivot.rolling("5min").mean()
    rolling.reset_index(inplace=True)
    return rolling


# -------------------------------------------------------
# 5. Export to CSV
# -------------------------------------------------------
def export_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved CSV → {filename}")


LOG_DIR = Path("logs/json")
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = Path("data/traffic.db")
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)   

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()

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