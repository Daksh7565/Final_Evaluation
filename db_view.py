# view_db.py
import pandas as pd
from pathlib import Path

from DB import SessionLocal
from DB import Route, RouteRun, VehicleTrack, CountEvent


EXPORT_DIR = Path("db_exports")
EXPORT_DIR.mkdir(exist_ok=True)


def export_to_csv(df, filename):
    """Save dataframe to CSV inside db_exports/ folder"""
    path = EXPORT_DIR / filename
    df.to_csv(path, index=False)
    print(f"✔ CSV saved → {path}")


def fetch_table(session, model, columns):
    """Generic function to convert table rows into dataframe"""
    rows = session.query(model).all()
    if not rows:
        return pd.DataFrame(columns=columns)

    data = []
    for row in rows:
        data.append({col: getattr(row, col) for col in columns})
    return pd.DataFrame(data)


def main():
    session = SessionLocal()

    print("\n=== EXPORTING DATABASE TABLES TO CSV ===")

    # ROUTES
    df_routes = fetch_table(
        session,
        Route,
        ["id", "name", "location", "line_config", "created_at"]
    )
    export_to_csv(df_routes, "routes.csv")

    # ROUTE RUNS
    df_runs = fetch_table(
        session,
        RouteRun,
        ["id", "route_id", "start_time", "end_time", "raw_video_path"]
    )
    export_to_csv(df_runs, "route_runs.csv")

    # VEHICLE TRACKS
    df_tracks = fetch_table(
        session,
        VehicleTrack,
        ["id", "run_id", "track_id", "class_name", "first_seen", "last_seen", "track_info"]
    )
    export_to_csv(df_tracks, "vehicle_tracks.csv")

    # COUNT EVENTS
    df_counts = fetch_table(
        session,
        CountEvent,
        ["id", "run_id", "route_id", "timestamp", "class_name", "count_increment"]
    )
    export_to_csv(df_counts, "count_events.csv")

    print("\n🎉 All CSV exports complete!")
    session.close()


if __name__ == "__main__":
    main()
