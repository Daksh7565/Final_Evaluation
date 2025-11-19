# # analytics.py
# from datetime import datetime, timedelta
# from sqlalchemy import func
# import pandas as pd

# from DB import SessionLocal
# from model import CountEvent


# # -------------------------------------------------------
# # 1. Per-minute aggregates
# # -------------------------------------------------------
# def get_per_minute_counts(route_id, run_id=None):
#     session = SessionLocal()

#     q = (
#         session.query(
#             func.strftime("%Y-%m-%d %H:%M", CountEvent.timestamp).label("minute"),
#             CountEvent.class_name,
#             func.sum(CountEvent.count_increment).label("count")
#         )
#         .filter(CountEvent.route_id == route_id)
#     )

#     if run_id:
#         q = q.filter(CountEvent.run_id == run_id)

#     q = q.group_by("minute", CountEvent.class_name).order_by("minute")

#     df = pd.DataFrame(q.all(), columns=["minute", "class_name", "count"])
#     session.close()
#     return df


# # -------------------------------------------------------
# # 2. Hourly aggregates
# # -------------------------------------------------------
# def get_hourly_counts(route_id):
#     session = SessionLocal()

#     q = (
#         session.query(
#             func.strftime("%Y-%m-%d %H", CountEvent.timestamp).label("hour"),
#             CountEvent.class_name,
#             func.sum(CountEvent.count_increment).label("count")
#         )
#         .filter(CountEvent.route_id == route_id)
#         .group_by("hour", CountEvent.class_name)
#         .order_by("hour")
#     )

#     df = pd.DataFrame(q.all(), columns=["hour", "class_name", "count"])
#     session.close()
#     return df


# # -------------------------------------------------------
# # 3. Class-wise distribution
# # -------------------------------------------------------
# def get_class_distribution(route_id):
#     session = SessionLocal()

#     q = (
#         session.query(
#             CountEvent.class_name,
#             func.sum(CountEvent.count_increment).label("count")
#         )
#         .filter(CountEvent.route_id == route_id)
#         .group_by(CountEvent.class_name)
#     )

#     df = pd.DataFrame(q.all(), columns=["class_name", "count"])
#     session.close()
#     return df


# # -------------------------------------------------------
# # 4. 5-minute rolling averages
# # -------------------------------------------------------
# def get_rolling_5min(df_per_minute):
#     if df_per_minute.empty:
#         return df_per_minute

#     df = df_per_minute.copy()
#     df["minute"] = pd.to_datetime(df["minute"])

#     # pivot into: minute | car | truck | bus
#     pivot = df.pivot_table(index="minute", columns="class_name", values="count", fill_value=0)

#     rolling = pivot.rolling("5min").mean()
#     rolling.reset_index(inplace=True)
#     return rolling


# # -------------------------------------------------------
# # 5. Export to CSV
# # -------------------------------------------------------
# def export_to_csv(df, filename):
#     df.to_csv(filename, index=False)
#     print(f"Saved CSV → {filename}")

# analytics.py
from datetime import datetime, timedelta
from sqlalchemy import func
import pandas as pd
import numpy as np

from DB import SessionLocal
from model import CountEvent


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
        return pd.DataFrame()

    df = df_per_minute.copy()
    df["minute"] = pd.to_datetime(df["minute"])

    # pivot into: minute | car | truck | bus
    pivot = df.pivot_table(index="minute", columns="class_name", values="count", fill_value=0)

    # Use '5T' for a 5-minute frequency window
    rolling = pivot.rolling("5T").mean()
    rolling.reset_index(inplace=True)
    return rolling


# -------------------------------------------------------
# 5. Export to CSV
# -------------------------------------------------------
def export_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved CSV → {filename}")


# =======================================================
# NEW & IMPROVED ANALYTICS FUNCTIONS
# =======================================================

# -------------------------------------------------------
# 6. Trend Score Calculation (NEW)
# -------------------------------------------------------
def get_trend_scores(df_per_minute, window_minutes=15):
    """
    Calculates the traffic trend score for each vehicle class over the last N minutes.
    The trend is the slope of a linear regression line fitted to the counts.
    - Positive slope: Traffic is increasing.
    - Negative slope: Traffic is decreasing.
    - Slope near zero: Traffic is stable.
    """
    if df_per_minute.empty or len(df_per_minute) < 2:
        return {}

    df = df_per_minute.copy()
    df["minute"] = pd.to_datetime(df["minute"])

    # Filter for the last N minutes of data
    cutoff_time = df["minute"].max() - timedelta(minutes=window_minutes)
    recent_df = df[df["minute"] >= cutoff_time]

    if len(recent_df) < 2:  # Need at least 2 points to fit a line
        return {}

    trend_scores = {}
    # Calculate trend for each vehicle class
    for class_name, group in recent_df.groupby("class_name"):
        if len(group) < 2:
            continue
        
        # Convert timestamps to a numerical format (seconds since the first point)
        # for regression analysis.
        time_as_numeric = (group["minute"] - group["minute"].min()).dt.total_seconds()
        counts = group["count"]
        
        # Fit a line (degree 1 polynomial) to the data points (time, count).
        # The first element of the result (m) is the slope of the line.
        try:
            m, _ = np.polyfit(time_as_numeric, counts, 1)
            trend_scores[class_name] = m
        except np.linalg.LinAlgError:
            trend_scores[class_name] = 0 # Handle cases with insufficient data for a stable fit

    return trend_scores


# -------------------------------------------------------
# 7. Peak Hour Analysis (ADVANCED)
# -------------------------------------------------------
def get_peak_hour(df_hourly):
    """Identifies the hour with the highest historical traffic count."""
    if df_hourly.empty:
        return None, 0
        
    # Sum counts per hour across all vehicle classes
    hourly_total = df_hourly.groupby('hour')['count'].sum()
    
    if hourly_total.empty:
        return None, 0
        
    # Find the hour with the maximum total count
    peak_hour_str = hourly_total.idxmax()
    peak_count = hourly_total.max()
    
    return peak_hour_str, peak_count


# -------------------------------------------------------
# 8. Anomaly Detection (ADVANCED)
# -------------------------------------------------------
def detect_anomalies(df_per_minute, std_dev_threshold=2.5, window='15T'):
    """
    Detects anomalous traffic spikes using a rolling standard deviation.
    An anomaly is a data point that is X standard deviations above the rolling mean.
    """
    if df_per_minute.empty or len(df_per_minute) < 2:
        return pd.DataFrame()
    
    df = df_per_minute.copy()
    df["minute"] = pd.to_datetime(df["minute"])
    pivot = df.pivot_table(index="minute", columns="class_name", values="count", fill_value=0)
    
    # Calculate rolling mean and standard deviation over a specified window
    rolling_mean = pivot.rolling(window=window).mean()
    rolling_std = pivot.rolling(window=window).std()

    # Define the anomaly threshold
    threshold = rolling_mean + (std_dev_threshold * rolling_std)
    
    # Identify data points that exceed the threshold
    is_anomaly = pivot > threshold
    
    # Return the rows from the pivot table that contain at least one anomaly
    anomalies = pivot[is_anomaly.any(axis=1)]
    return anomalies