# # route_runner.py
# import os
# from pathlib import Path
# from datetime import datetime, timezone
# from analysis import (get_per_minute_counts,get_hourly_counts,get_class_distribution,get_rolling_5min,export_to_csv)

# import cv2
# import numpy as np

# from rfdetr import RFDETRBase
# from sort.Tracker import Sort

# from DB import engine, SessionLocal
# from model import Base, Route, RouteRun, VehicleTrack, CountEvent

# from json_logging import log_frame_json

# # create DB tables if not exist
# Base.metadata.create_all(bind=engine)
# def run_analytics(route_id):
#         df_min = get_per_minute_counts(route_id)
#         df_hour = get_hourly_counts(route_id)
#         df_dist = get_class_distribution(route_id)
#         df_roll = get_rolling_5min(df_min)

#         export_to_csv(df_min, "analytics/per_minute.csv")
#         export_to_csv(df_hour, "analytics/hourly.csv")
#         export_to_csv(df_dist, "analytics/class_distribution.csv")
#         export_to_csv(df_roll, "analytics/rolling_5min.csv")


# # CONFIG: set your routes (5 routes)
# ROUTE_VIDEOS = {
#     "route_01": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4",
#     "route_02": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Cars Moving On Road Stock Footage - Free Download.mp4",
#     "route_03": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Traffic IP Camera video.mp4",
#     "route_04": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\27260-362770008_large.mp4",
#     "route_05": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-534232220-640_adpp_is.mp4",
#     "route_06": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-866517852-640_adpp_is.mp4"
# }

# CONFIDENCE_THRESHOLD = 0.5

# def get_or_create_route(session, name, location=None, line_config=None):
#     route = session.query(Route).filter_by(name=name).first()
#     if route:
#         return route
#     route = Route(name=name, location=location, line_config=line_config)
#     session.add(route)
#     session.commit()
#     session.refresh(route)
#     return route

# def start_route_run(session, route: Route, video_path: str):
#     run = RouteRun(route_id=route.id, raw_video_path=video_path, start_time=datetime.utcnow())
#     session.add(run)
#     session.commit()
#     session.refresh(run)
#     return run

# def end_route_run(session, run: RouteRun):
#     run.end_time = datetime.utcnow()
#     session.commit()

# def upsert_vehicle_track(session, run_id, track_id, class_name, first_seen, last_seen, metadata=None):
#     # Attempt to find existing track record for this run+track_id+class
#     vt = session.query(VehicleTrack).filter_by(run_id=run_id, track_id=track_id, class_name=class_name).first()
#     if vt is None:
#         vt = VehicleTrack(run_id=run_id, track_id=track_id, class_name=class_name,
#                           first_seen=first_seen, last_seen=last_seen, metadata=metadata or {})
#         session.add(vt)
#     else:
#         vt.last_seen = last_seen
#         # optionally append bbox history in metadata outside for brevity
#     session.commit()

# def insert_count_event(session, run_id, route_id, class_name, increment=1):
#     ce = CountEvent(run_id=run_id, route_id=route_id, class_name=class_name, count_increment=increment, timestamp=datetime.utcnow())
#     session.add(ce)
#     session.commit()

# def process_route(route_name, video_path, model, tracker, session, route_obj):
#     print(f"Processing {route_name} -> {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Could not open {video_path}")
#         return

#     run = start_route_run(session, route_obj, video_path)

#     # counters and history
#     counts = {"car": 0, "truck": 0, "bus": 0}
#     track_history = {}  # track_id -> last centroid y

#     frame_index = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_index += 1
#         frame_h, frame_w = frame.shape[:2]
#         line_y = int(frame_h * 0.80)  # 80% height as requested

#         # detect
#         results = model.predict(frame, threshold=CONFIDENCE_THRESHOLD)
#         # model.class_names may be dict {id: name}
#         class_names = results.class_names if hasattr(results, "class_names") else model.class_names
#         # results: compatible with previous code (xyxy, class_id, confidence)

#         # filter
#         target_classes = ["car", "truck", "bus"]
#         # handle class_names dict/list:
#         if isinstance(class_names, dict):
#             target_ids = [cid for cid, n in class_names.items() if n in target_classes]
#         else:
#             target_ids = [class_names.index(c) for c in target_classes if c in class_names]

#         mask = np.isin(results.class_id, target_ids)
#         filtered = results[mask]

#         if len(filtered) > 0:
#             dets = np.hstack((filtered.xyxy, filtered.confidence.reshape(-1, 1)))
#         else:
#             dets = np.empty((0, 5))

#         tracked_objects = tracker.update(dets)

#         # prepare per-frame detection records for JSON
#         detections_log = []
#         counts_incremented = {"car": 0, "truck": 0, "bus": 0}

#         # For each tracked object: draw logic + decide crossing
#         for x1, y1, x2, y2, tid in tracked_objects:
#             tid = int(tid)
#             x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
#             cx = int((x1_i + x2_i) / 2)
#             cy = int((y1_i + y2_i) / 2)

#             # find matching detection to get class
#             cls_name = None
#             if len(filtered) > 0:
#                 # smallest distance between centers / left x coordinate
#                 dists = np.abs(filtered.xyxy[:, 0] - x1)
#                 idx = int(np.argmin(dists))
#                 cls_name = class_names[filtered.class_id[idx]] if isinstance(class_names, dict) else class_names[filtered.class_id[idx]]

#             if cls_name is None:
#                 cls_name = "unknown"

#             # track history
#             prev_y = track_history.get(tid, None)
#             # check crossing downward
#             crossed = False
#             if prev_y is not None and prev_y < line_y <= cy:
#                 crossed = True
#                 if cls_name in counts:
#                     counts[cls_name] += 1
#                     counts_incremented[cls_name] += 1
#                     # write count event to DB
#                     insert_count_event(session, run.id, route_obj.id, cls_name, increment=1)
#             # update track history
#             track_history[tid] = cy

#             # upsert track meta
#             now = datetime.utcnow()
#             upsert_vehicle_track(session, run.id, tid, cls_name, first_seen=now, last_seen=now, metadata={"bbox":[x1_i,y1_i,x2_i,y2_i]})

#             detections_log.append({
#                 "track_id": tid,
#                 "class": cls_name,
#                 "bbox": [int(x1_i), int(y1_i), int(x2_i), int(y2_i)],
#                 "centroid": [cx, cy],
#                 "crossed_line": crossed
#             })

#         # write per-frame JSON
#         log_frame_json(route_name, frame_index, detections_log, counts_incremented)

#     # finalize run
#     cap.release()
#     end_route_run(session, run)
#     print(f"Finished {route_name}. Totals: {counts}")

# def main():
#     # initialize model and tracker once
#     print("Loading model...")
#     model = RFDETRBase()
#     tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
#     print("Model + tracker ready.")

#     session = SessionLocal()

#     # ensure route entries exist in DB
#     for rname, vpath in ROUTE_VIDEOS.items():
#         route_obj = get_or_create_route(session, rname, location=vpath, line_config={"type":"horizontal","y_percent":0.8})
#         # make sure video path exists - if not warn but continue
#         if not Path(vpath).exists():
#             print(f"Warning: {vpath} does not exist. Place the file at this path or update ROUTE_VIDEOS.")
#             continue
#         process_route(rname, vpath, model, tracker, session, route_obj)

#     session.close()
#     print("All routes processed.")
    
    
# if __name__ == "__main__":
#     main()


# # route_runner.py
# import os
# from pathlib import Path
# from datetime import datetime, timezone
# from analysis import (get_per_minute_counts,get_hourly_counts,get_class_distribution,get_rolling_5min,export_to_csv)

# import cv2
# import numpy as np

# from rfdetr import RFDETRBase
# from sort.Tracker import Sort

# from DB import engine, SessionLocal
# from model import Base, Route, RouteRun, VehicleTrack, CountEvent

# from json_logging import log_frame_json

# # create DB tables if not exist
# Base.metadata.create_all(bind=engine)
# def run_analytics(route_id):
#         df_min = get_per_minute_counts(route_id)
#         df_hour = get_hourly_counts(route_id)
#         df_dist = get_class_distribution(route_id)
#         df_roll = get_rolling_5min(df_min)

#         export_to_csv(df_min, "analytics/per_minute.csv")
#         export_to_csv(df_hour, "analytics/hourly.csv")
#         export_to_csv(df_dist, "analytics/class_distribution.csv")
#         export_to_csv(df_roll, "analytics/rolling_5min.csv")


# # CONFIG: set your routes
# ROUTE_VIDEOS = {
#     "route_01": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4",
#     "route_02": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Cars Moving On Road Stock Footage - Free Download.mp4",
#     "route_03": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Traffic IP Camera video.mp4",
#     "route_04": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\27260-362770008_large.mp4",
#     "route_05": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-534232220-640_adpp_is.mp4",
#     "route_06": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-866517852-640_adpp_is.mp4"
# }

# CONFIDENCE_THRESHOLD = 0.5

# def get_or_create_route(session, name, location=None, line_config=None):
#     route = session.query(Route).filter_by(name=name).first()
#     if route:
#         return route
#     route = Route(name=name, location=location, line_config=line_config)
#     session.add(route)
#     session.commit()
#     session.refresh(route)
#     return route

# def start_route_run(session, route: Route, video_path: str):
#     run = RouteRun(route_id=route.id, raw_video_path=video_path, start_time=datetime.utcnow())
#     session.add(run)
#     session.commit()
#     session.refresh(run)
#     return run

# def end_route_run(session, run: RouteRun):
#     run.end_time = datetime.utcnow()
#     session.commit()

# def upsert_vehicle_track(session, run_id, track_id, class_name, first_seen, last_seen, metadata=None):
#     # Attempt to find existing track record for this run+track_id+class
#     vt = session.query(VehicleTrack).filter_by(run_id=run_id, track_id=track_id, class_name=class_name).first()
#     if vt is None:
#         vt = VehicleTrack(run_id=run_id, track_id=track_id, class_name=class_name,
#                           first_seen=first_seen, last_seen=last_seen, metadata=metadata or {})
#         session.add(vt)
#     else:
#         vt.last_seen = last_seen
#         # optionally append bbox history in metadata outside for brevity
#     session.commit()

# def insert_count_event(session, run_id, route_id, class_name, increment=1):
#     ce = CountEvent(run_id=run_id, route_id=route_id, class_name=class_name, count_increment=increment, timestamp=datetime.utcnow())
#     session.add(ce)
#     session.commit()

# def process_route(route_name, video_path, model, tracker, session, route_obj):
#     print(f"Processing {route_name} -> {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Could not open {video_path}")
#         return

#     run = start_route_run(session, route_obj, video_path)

#     # counters and history
#     counts = {"car": 0, "truck": 0, "bus": 0}
#     track_history = {}  # track_id -> last centroid y

#     frame_index = 0
#     window_name = f"Processing: {route_name}"

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_index += 1
#         frame_h, frame_w = frame.shape[:2]
#         line_y = int(frame_h * 0.80)  # 80% height as requested

#         # detect
#         results = model.predict(frame, threshold=CONFIDENCE_THRESHOLD)
#         class_names = results.class_names if hasattr(results, "class_names") else model.class_names
        
#         # filter
#         target_classes = ["car", "truck", "bus"]
#         if isinstance(class_names, dict):
#             target_ids = [cid for cid, n in class_names.items() if n in target_classes]
#         else:
#             target_ids = [class_names.index(c) for c in class_names if c in class_names]

#         mask = np.isin(results.class_id, target_ids)
#         filtered = results[mask]

#         if len(filtered) > 0:
#             dets = np.hstack((filtered.xyxy, filtered.confidence.reshape(-1, 1)))
#         else:
#             dets = np.empty((0, 5))

#         tracked_objects = tracker.update(dets)

#         detections_log = []
#         counts_incremented = {"car": 0, "truck": 0, "bus": 0}
        
#         # --- START: VISUALIZATION CODE ---
#         # Draw the counting line
#         cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 2)
#         # --- END: VISUALIZATION CODE ---
        

#         for x1, y1, x2, y2, tid in tracked_objects:
#             tid = int(tid)
#             x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
#             cx = int((x1_i + x2_i) / 2)
#             cy = int((y1_i + y2_i) / 2)

#             cls_name = "unknown"
#             if len(filtered) > 0:
#                 dists = np.abs(filtered.xyxy[:, 0] - x1)
#                 idx = int(np.argmin(dists))
#                 cls_name = class_names[filtered.class_id[idx]] if isinstance(class_names, dict) else class_names[filtered.class_id[idx]]

#             prev_y = track_history.get(tid, None)
#             crossed = False
#             if prev_y is not None and prev_y < line_y <= cy:
#                 crossed = True
#                 if cls_name in counts:
#                     counts[cls_name] += 1
#                     counts_incremented[cls_name] += 1
#                     insert_count_event(session, run.id, route_obj.id, cls_name, increment=1)
            
#             track_history[tid] = cy

#             now = datetime.utcnow()
#             upsert_vehicle_track(session, run.id, tid, cls_name, first_seen=now, last_seen=now, metadata={"bbox":[x1_i,y1_i,x2_i,y2_i]})

#             detections_log.append({
#                 "track_id": tid, "class": cls_name, "bbox": [x1_i, y1_i, x2_i, y2_i],
#                 "centroid": [cx, cy], "crossed_line": crossed
#             })
            
#             # --- START: VISUALIZATION CODE ---
#             # Draw bounding box and label for each tracked object
#             color = (0, 255, 0) if not crossed else (0, 0, 255) # Green for normal, Red after crossing
#             label = f"{cls_name} ID:{tid}"
#             cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
#             cv2.putText(frame, label, (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#             # --- END: VISUALIZATION CODE ---

#         log_frame_json(route_name, frame_index, detections_log, counts_incremented)

#         # --- START: VISUALIZATION CODE ---
#         # Display the live counts on the top-left corner
#         y_offset = 40
#         for v_class, count in counts.items():
#             text = f"{v_class.title()}: {count}"
#             cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             y_offset += 40

#         # Show the frame in a window
#         cv2.imshow(window_name, frame)

#         # Allow user to quit by pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         # --- END: VISUALIZATION CODE ---

#     # finalize run
#     cap.release()
#     cv2.destroyAllWindows() # Close the window when the video finishes or user quits
#     end_route_run(session, run)
#     print(f"Finished {route_name}. Totals: {counts}")

# def main():
#     print("Loading model...")
#     model = RFDETRBase()
#     tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
#     print("Model + tracker ready.")

#     session = SessionLocal()

#     for rname, vpath in ROUTE_VIDEOS.items():
#         route_obj = get_or_create_route(session, rname, location=vpath, line_config={"type":"horizontal","y_percent":0.8})
#         if not Path(vpath).exists():
#             print(f"Warning: {vpath} does not exist. Place the file at this path or update ROUTE_VIDEOS.")
#             continue
#         process_route(rname, vpath, model, tracker, session, route_obj)

#     session.close()
#     print("All routes processed.")
    
    
# if __name__ == "__main__":
#     main()

# route_runner.py
import os
import time  # Import the time library for performance measurement
from pathlib import Path
from datetime import datetime, timezone
from analysis import (get_per_minute_counts,get_hourly_counts,get_class_distribution,get_rolling_5min,export_to_csv)

import cv2
import numpy as np

from rfdetr import RFDETRBase
from sort.Tracker import Sort

from DB import engine, SessionLocal
from model import Base, Route, RouteRun, VehicleTrack, CountEvent

from json_logging import log_frame_json

# create DB tables if not exist
Base.metadata.create_all(bind=engine)
def run_analytics(route_id):
        df_min = get_per_minute_counts(route_id)
        df_hour = get_hourly_counts(route_id)
        df_dist = get_class_distribution(route_id)
        df_roll = get_rolling_5min(df_min)

        export_to_csv(df_min, "analytics/per_minute.csv")
        export_to_csv(df_hour, "analytics/hourly.csv")
        export_to_csv(df_dist, "analytics/class_distribution.csv")
        export_to_csv(df_roll, "analytics/rolling_5min.csv")


# CONFIG: set your routes
ROUTE_VIDEOS = {
    "route_01": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4",
    "route_02": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Cars Moving On Road Stock Footage - Free Download.mp4",
    "route_03": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Traffic IP Camera video.mp4",
    "route_04": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\27260-362770008_large.mp4",
    "route_05": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-534232220-640_adpp_is.mp4",
    "route_06": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-866517852-640_adpp_is.mp4",
    "route_07": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\2103099-uhd_3840_2160_30fps (1).mp4",
    "route_08": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-1282097063-640_adpp_is.mp4"
}

CONFIDENCE_THRESHOLD = 0.5

def get_or_create_route(session, name, location=None, line_config=None):
    route = session.query(Route).filter_by(name=name).first()
    if route:
        return route
    route = Route(name=name, location=location, line_config=line_config)
    session.add(route)
    session.commit()
    session.refresh(route)
    return route

def start_route_run(session, route: Route, video_path: str):
    run = RouteRun(route_id=route.id, raw_video_path=video_path, start_time=datetime.utcnow())
    session.add(run)
    session.commit()
    session.refresh(run)
    return run

def end_route_run(session, run: RouteRun):
    run.end_time = datetime.utcnow()
    session.commit()

def upsert_vehicle_track(session, run_id, track_id, class_name, first_seen, last_seen, metadata=None):
    # Attempt to find existing track record for this run+track_id+class
    vt = session.query(VehicleTrack).filter_by(run_id=run_id, track_id=track_id, class_name=class_name).first()
    if vt is None:
        vt = VehicleTrack(run_id=run_id, track_id=track_id, class_name=class_name,
                          first_seen=first_seen, last_seen=last_seen, metadata=metadata or {})
        session.add(vt)
    else:
        vt.last_seen = last_seen
        # optionally append bbox history in metadata outside for brevity
    session.commit()

def insert_count_event(session, run_id, route_id, class_name, increment=1):
    ce = CountEvent(run_id=run_id, route_id=route_id, class_name=class_name, count_increment=increment, timestamp=datetime.utcnow())
    session.add(ce)
    session.commit()

def process_route(route_name, video_path, model, tracker, session, route_obj):
    print(f"Processing {route_name} -> {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    run = start_route_run(session, route_obj, video_path)

    # counters and history
    counts = {"car": 0, "truck": 0, "bus": 0}
    track_history = {}  # track_id -> last centroid y

    frame_index = 0
    window_name = f"Processing: {route_name}"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- START: LATENCY MEASUREMENT ---
        start_time = time.time()
        # --- END: LATENCY MEASUREMENT ---

        frame_index += 1
        frame_h, frame_w = frame.shape[:2]
        line_y = int(frame_h * 0.80)

        # detect
        results = model.predict(frame, threshold=CONFIDENCE_THRESHOLD)
        class_names = results.class_names if hasattr(results, "class_names") else model.class_names
        
        # filter
        target_classes = ["car", "truck", "bus"]
        if isinstance(class_names, dict):
            target_ids = [cid for cid, n in class_names.items() if n in target_classes]
        else:
            target_ids = [class_names.index(c) for c in target_classes if c in class_names]

        mask = np.isin(results.class_id, target_ids)
        filtered = results[mask]

        if len(filtered) > 0:
            dets = np.hstack((filtered.xyxy, filtered.confidence.reshape(-1, 1)))
        else:
            dets = np.empty((0, 5))

        tracked_objects = tracker.update(dets)

        detections_log = []
        counts_incremented = {"car": 0, "truck": 0, "bus": 0}
        
        # Draw the counting line
        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 2)
        
        for x1, y1, x2, y2, tid in tracked_objects:
            tid = int(tid)
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            cx = int((x1_i + x2_i) / 2)
            cy = int((y1_i + y2_i) / 2)

            cls_name = "unknown"
            if len(filtered) > 0:
                dists = np.abs(filtered.xyxy[:, 0] - x1)
                idx = int(np.argmin(dists))
                cls_name = class_names[filtered.class_id[idx]] if isinstance(class_names, dict) else class_names[filtered.class_id[idx]]

            prev_y = track_history.get(tid, None)
            crossed = False
            if prev_y is not None and prev_y < line_y <= cy:
                crossed = True
                if cls_name in counts:
                    counts[cls_name] += 1
                    counts_incremented[cls_name] += 1
                    insert_count_event(session, run.id, route_obj.id, cls_name, increment=1)
            
            track_history[tid] = cy

            now = datetime.utcnow()
            upsert_vehicle_track(session, run.id, tid, cls_name, first_seen=now, last_seen=now, metadata={"bbox":[x1_i,y1_i,x2_i,y2_i]})

            detections_log.append({
                "track_id": tid, "class": cls_name, "bbox": [x1_i, y1_i, x2_i, y2_i],
                "centroid": [cx, cy], "crossed_line": crossed
            })
            
            color = (0, 255, 0) if not crossed else (0, 0, 255)
            label = f"{cls_name} ID:{tid}"
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
            cv2.putText(frame, label, (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        log_frame_json(route_name, frame_index, detections_log, counts_incremented)

        # Display the live counts on the top-left corner
        y_offset = 40
        for v_class, count in counts.items():
            text = f"{v_class.title()}: {count}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 40

        # --- START: LATENCY CALCULATION AND DISPLAY ---
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        latency_text = f"Latency: {processing_time_ms:.2f} ms"
        fps_text = f"FPS: {fps:.2f}"

        # Display latency and FPS on the top-right corner
        cv2.putText(frame, latency_text, (frame_w - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, fps_text, (frame_w - 320, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # --- END: LATENCY CALCULATION AND DISPLAY ---

        # Show the frame in a window
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    end_route_run(session, run)
    print(f"Finished {route_name}. Totals: {counts}")

def main():
    print("Loading model...")
    model = RFDETRBase()
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
    print("Model + tracker ready.")

    session = SessionLocal()

    for rname, vpath in ROUTE_VIDEOS.items():
        route_obj = get_or_create_route(session, rname, location=vpath, line_config={"type":"horizontal","y_percent":0.8})
        if not Path(vpath).exists():
            print(f"Warning: {vpath} does not exist. Place the file at this path or update ROUTE_VIDEOS.")
            continue
        process_route(rname, vpath, model, tracker, session, route_obj)

    session.close()
    print("All routes processed.")
    
    
if __name__ == "__main__":
    main()