# models.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

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
