# import os
# import time
# from pathlib import Path
# from datetime import datetime
# from collections import defaultdict

# import cv2
# import numpy as np
# import torch

# from analysis import (
#     get_per_minute_counts, get_hourly_counts, get_class_distribution, get_rolling_5min, export_to_csv
# )

# from rfdetr import RFDETRBase
# from sort.Tracker import Sort

# from DB import engine, SessionLocal
# from model import Base, Route, RouteRun, VehicleTrack, CountEvent

# from json_logging import log_frame_json

# # create DB tables if not exist
# Base.metadata.create_all(bind=engine)

# # CONFIG
# ROUTE_VIDEOS = {
#     "route_01": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4",
#     "route_02": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Cars Moving On Road Stock Footage - Free Download.mp4",
#     "route_03": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Traffic IP Camera video.mp4",
#     "route_04": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\27260-362770008_large.mp4",
#     "route_05": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-534232220-640_adpp_is.mp4",
#     "route_06": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-866517852-640_adpp_is.mp4",
#     "route_07": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\2103099-uhd_3840_2160_30fps (1).mp4",
#     "route_08": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-1282097063-640_adpp_is.mp4"
# }

# CONFIDENCE_THRESHOLD = 0.5

# # Performance tuning parameters
# INFERENCE_SIZE = (960, 540)        # resize before inference (w, h) — smaller = faster
# DETECTION_INTERVAL = 2             # run detector every N frames (use tracker between detections)
# BATCH_COMMIT_SIZE = 20             # commit DB writes every N frames
# SHOW_EVERY_N = 2                   # only draw/display every N frames (reduce imshow overhead)
# LOG_JSON_EVERY = 5                 # write JSON logs every N frames
# REPORT_INTERVAL = 10               # print latency every N frames (detailed print)

# TARGET_CLASSES = ["car", "truck", "bus"]


# def run_analytics(route_id):
#     df_min = get_per_minute_counts(route_id)
#     df_hour = get_hourly_counts(route_id)
#     df_dist = get_class_distribution(route_id)
#     df_roll = get_rolling_5min(df_min)

#     export_to_csv(df_min, "analytics/per_minute.csv")
#     export_to_csv(df_hour, "analytics/hourly.csv")
#     export_to_csv(df_dist, "analytics/class_distribution.csv")
#     export_to_csv(df_roll, "analytics/rolling_5min.csv")


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


# def process_route(route_name, video_path, model, tracker, session, route_obj):
#     """
#     Optimized per-route processing:
#     - Resize input before inference
#     - Run detection every DETECTION_INTERVAL frames
#     - Batch DB writes for counts and vehicle tracks
#     - Reduce visualization frequency
#     - Print detailed latency (detect, track, total, fps) every REPORT_INTERVAL frames
#     """
#     print(f"Processing {route_name} -> {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Could not open {video_path}")
#         return

#     run = start_route_run(session, route_obj, video_path)

#     # counters and history
#     counts = {c: 0 for c in TARGET_CLASSES}
#     track_history = {}  # track_id -> last centroid y

#     frame_index = 0
#     window_name = f"Processing: {route_name}"

#     # batch buffers for DB writes
#     pending_count_events = []
#     pending_vehicle_updates = {}  # key = (track_id, class_name) -> {"first_seen": ts, "last_seen": ts, "metadata": ...}
#     pending_json_logs = []

#     prev_results = None
#     class_names = None

#     # latency accumulators (for printing)
#     detect_time_ms = 0.0
#     track_time_ms = 0.0
#     total_time_ms = 0.0

#     # helper: scale boxes back to original frame size
#     def scale_boxes_xyxy(boxes, scale_x, scale_y):
#         boxes[:, 0] *= scale_x
#         boxes[:, 1] *= scale_y
#         boxes[:, 2] *= scale_x
#         boxes[:, 3] *= scale_y
#         return boxes

#     # pre-check GPU usage and model device
#     using_cuda = torch.cuda.is_available()
#     model_device = None
#     try:
#         # try to get model device from parameters
#         if hasattr(model, "parameters"):
#             p = next(model.parameters(), None)
#             if p is not None:
#                 model_device = p.device
#     except Exception:
#         model_device = None

#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             loop_start = time.time()

#             frame_index += 1
#             frame_h, frame_w = frame.shape[:2]
#             line_y = int(frame_h * 0.80)

#             # --- Decide whether to run detector this frame ---
#             do_detect = (frame_index % DETECTION_INTERVAL == 0) or (prev_results is None)

#             # Resize for inference (faster)
#             inp_w, inp_h = INFERENCE_SIZE
#             small = cv2.resize(frame, (inp_w, inp_h))
#             scale_x = frame_w / inp_w
#             scale_y = frame_h / inp_h

#             results = None
#             detect_time_ms = 0.0

#             # Run detection (GPU if available)
#             if do_detect:
#                 detect_start = time.time()
#                 if using_cuda:
#                     try:
#                         torch.cuda.synchronize()
#                     except Exception:
#                         pass
#                 # Use inference_mode to reduce overhead
#                 with torch.inference_mode():
#                     results = model.predict(small, threshold=CONFIDENCE_THRESHOLD)
#                 if using_cuda:
#                     try:
#                         torch.cuda.synchronize()
#                     except Exception:
#                         pass
#                 detect_time_ms = (time.time() - detect_start) * 1000.0

#                 # update class_names mapping/array for later use
#                 class_names = results.class_names if hasattr(results, "class_names") else getattr(model, "class_names", None)

#                 # Filter to target classes
#                 if isinstance(class_names, dict):
#                     target_ids = [cid for cid, n in class_names.items() if n in TARGET_CLASSES]
#                 else:
#                     target_ids = [class_names.index(c) for c in TARGET_CLASSES if class_names and c in class_names]

#                 # Try to get arrays from results
#                 try:
#                     class_id_arr = np.array(results.class_id)
#                 except Exception:
#                     # fallback: if results.class_id is torch tensor
#                     try:
#                         class_id_arr = np.array(results.class_id.cpu())
#                     except Exception:
#                         class_id_arr = np.array([])

#                 # mask detections by target classes
#                 if class_id_arr.size:
#                     mask = np.isin(class_id_arr, target_ids)
#                     try:
#                         filtered = results[mask]
#                     except Exception:
#                         filtered = results
#                 else:
#                     mask = np.array([], dtype=bool)
#                     filtered = results

#                 # Extract xyxy/conf/class arrays safely
#                 xyxy = np.empty((0, 4))
#                 conf = np.empty((0, 1))
#                 class_ids = np.empty((0,), dtype=int)
#                 try:
#                     if hasattr(filtered, "xyxy"):
#                         xyxy = np.array(filtered.xyxy).astype(float)
#                     if hasattr(filtered, "confidence"):
#                         conf = np.array(filtered.confidence).astype(float).reshape(-1, 1)
#                     if hasattr(filtered, "class_id"):
#                         class_ids = np.array(filtered.class_id).astype(int)
#                 except Exception:
#                     # keep arrays empty on failure
#                     pass

#                 # scale boxes back to original resolution
#                 if xyxy.size:
#                     xyxy = scale_boxes_xyxy(xyxy.copy(), scale_x, scale_y)

#                 # prepare dets for tracker: x1,y1,x2,y2,score
#                 if xyxy.size:
#                     dets = np.hstack((xyxy, conf))
#                 else:
#                     dets = np.empty((0, 5))

#                 prev_results = {
#                     "xyxy": xyxy,
#                     "conf": conf,
#                     "class_id": class_ids,
#                     "dets": dets,
#                     "class_names": class_names
#                 }
#             else:
#                 # reuse previous detections (no model call)
#                 if prev_results is None:
#                     prev_results = {"xyxy": np.empty((0, 4)), "conf": np.empty((0, 1)), "class_id": np.array([]), "dets": np.empty((0, 5)), "class_names": class_names or {}}

#             # Tracking timing
#             track_start = time.time()
#             try:
#                 tracked_objects = tracker.update(prev_results["dets"])
#             except Exception:
#                 # if tracker fails, continue with zero detections
#                 tracked_objects = np.empty((0, 5))
#             track_time_ms = (time.time() - track_start) * 1000.0

#             detections_log_frame = []
#             counts_incremented = {c: 0 for c in TARGET_CLASSES}

#             # Per-tracked-object processing (no DB commits here)
#             for x1, y1, x2, y2, tid in tracked_objects:
#                 tid = int(tid)
#                 x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
#                 cx = int((x1_i + x2_i) / 2)
#                 cy = int((y1_i + y2_i) / 2)

#                 # match to the nearest detection to get class (if any)
#                 cls_name = "unknown"
#                 if prev_results["xyxy"].size:
#                     try:
#                         dists = np.abs(prev_results["xyxy"][:, 0] - x1)
#                         idx = int(np.argmin(dists))
#                         cls_id = int(prev_results["class_id"][idx]) if prev_results["class_id"].size else None
#                         if cls_id is not None:
#                             if isinstance(prev_results["class_names"], dict):
#                                 cls_name = prev_results["class_names"].get(cls_id, "unknown")
#                             else:
#                                 try:
#                                     cls_name = prev_results["class_names"][cls_id]
#                                 except Exception:
#                                     cls_name = "unknown"
#                     except Exception:
#                         cls_name = "unknown"

#                 prev_y = track_history.get(tid, None)
#                 crossed = False
#                 if prev_y is not None and prev_y < line_y <= cy:
#                     crossed = True
#                     if cls_name in counts:
#                         counts[cls_name] += 1
#                         counts_incremented[cls_name] += 1
#                         # queue count event (batch commit later)
#                         pending_count_events.append(
#                             CountEvent(run_id=run.id, route_id=route_obj.id, class_name=cls_name, count_increment=1, timestamp=datetime.utcnow())
#                         )

#                 # update track_history
#                 track_history[tid] = cy

#                 # prepare vehicle track upsert in pending dict
#                 key = (tid, cls_name)
#                 now = datetime.utcnow()
#                 existing = pending_vehicle_updates.get(key)
#                 if existing:
#                     # update last_seen and metadata
#                     existing["last_seen"] = now
#                     existing["metadata"] = {"bbox": [x1_i, y1_i, x2_i, y2_i]}
#                 else:
#                     pending_vehicle_updates[key] = {
#                         "run_id": run.id,
#                         "track_id": tid,
#                         "class_name": cls_name,
#                         "first_seen": now,
#                         "last_seen": now,
#                         "metadata": {"bbox": [x1_i, y1_i, x2_i, y2_i]},
#                     }

#                 detections_log_frame.append({
#                     "track_id": tid,
#                     "class": cls_name,
#                     "bbox": [x1_i, y1_i, x2_i, y2_i],
#                     "centroid": [cx, cy],
#                     "crossed_line": crossed
#                 })

#             # add to pending JSON logs (batch write)
#             if frame_index % LOG_JSON_EVERY == 0:
#                 # write now
#                 try:
#                     log_frame_json(route_name, frame_index, detections_log_frame, counts_incremented)
#                 except Exception:
#                     # ignore logging errors to avoid crashing loop
#                     pass
#             else:
#                 pending_json_logs.append((route_name, frame_index, detections_log_frame, counts_incremented))
#                 # flush buffer if very large
#                 if len(pending_json_logs) > 200:
#                     for args in pending_json_logs:
#                         try:
#                             log_frame_json(*args)
#                         except Exception:
#                             pass
#                     pending_json_logs = []

#             # Visualization (draw & imshow less frequently)
#             do_draw = (frame_index % SHOW_EVERY_N == 0)
#             if do_draw:
#                 try:
#                     # draw counting line
#                     cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 2)

#                     for det in detections_log_frame:
#                         x1_i, y1_i, x2_i, y2_i = det["bbox"]
#                         tid = det["track_id"]
#                         cls_name = det["class"]
#                         crossed = det["crossed_line"]
#                         color = (0, 255, 0) if not crossed else (0, 0, 255)
#                         label = f"{cls_name} ID:{tid}"
#                         cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
#                         cv2.putText(frame, label, (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#                     # show counts
#                     y_offset = 40
#                     for v_class, count in counts.items():
#                         text = f"{v_class.title()}: {count}"
#                         cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                         y_offset += 40

#                     cv2.imshow(window_name, frame)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#                 except Exception:
#                     # continue on drawing errors
#                     pass

#             # BATCH DB COMMIT: every BATCH_COMMIT_SIZE frames, flush pending events & upserts
#             if frame_index % BATCH_COMMIT_SIZE == 0:
#                 if pending_count_events or pending_vehicle_updates:
#                     try:
#                         # Vehicle upsert logic:
#                         keys = list(pending_vehicle_updates.keys())
#                         track_ids = [k[0] for k in keys]
#                         # Fetch existing tracks for this run
#                         existing_tracks = session.query(VehicleTrack).filter(
#                             VehicleTrack.run_id == run.id,
#                             VehicleTrack.track_id.in_(track_ids)
#                         ).all() if track_ids else []

#                         existing_map = {(vt.track_id, vt.class_name): vt for vt in existing_tracks}

#                         # Prepare list of new VehicleTrack objects to bulk insert
#                         to_insert = []
#                         for key, val in pending_vehicle_updates.items():
#                             tid, cls = key
#                             vt = existing_map.get((tid, cls))
#                             if vt:
#                                 # update existing row
#                                 vt.last_seen = val["last_seen"]
#                                 vt.metadata = val["metadata"]
#                             else:
#                                 to_insert.append(VehicleTrack(
#                                     run_id=val["run_id"],
#                                     track_id=val["track_id"],
#                                     class_name=val["class_name"],
#                                     first_seen=val["first_seen"],
#                                     last_seen=val["last_seen"],
#                                     metadata=val["metadata"]
#                                 ))

#                         # Bulk save new objects
#                         if to_insert:
#                             session.bulk_save_objects(to_insert)

#                         # Bulk save count events
#                         if pending_count_events:
#                             session.bulk_save_objects(pending_count_events)

#                         session.commit()
#                     except Exception as e:
#                         print("Error during batch DB commit:", e)
#                     finally:
#                         pending_count_events = []
#                         pending_vehicle_updates = {}

#             # compute total loop time and print latency if needed
#             total_time_ms = (time.time() - loop_start) * 1000.0
#             if frame_index % REPORT_INTERVAL == 0:
#                 fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
#                 print(f"[{route_name}] Frame {frame_index} | Detect: {detect_time_ms:.2f} ms | Track: {track_time_ms:.2f} ms | Total: {total_time_ms:.2f} ms | FPS: {fps:.2f}")

#         # end while cap
#     finally:
#         # flush any remaining pending operations
#         if pending_json_logs:
#             for args in pending_json_logs:
#                 try:
#                     log_frame_json(*args)
#                 except Exception:
#                     pass
#             pending_json_logs = []

#         if pending_count_events or pending_vehicle_updates:
#             # attempt one final commit
#             try:
#                 keys = list(pending_vehicle_updates.keys())
#                 track_ids = [k[0] for k in keys]
#                 existing_tracks = session.query(VehicleTrack).filter(
#                     VehicleTrack.run_id == run.id,
#                     VehicleTrack.track_id.in_(track_ids)
#                 ).all() if track_ids else []
#                 existing_map = {(vt.track_id, vt.class_name): vt for vt in existing_tracks}
#                 to_insert = []
#                 for key, val in pending_vehicle_updates.items():
#                     tid, cls = key
#                     vt = existing_map.get((tid, cls))
#                     if vt:
#                         vt.last_seen = val["last_seen"]
#                         vt.metadata = val["metadata"]
#                     else:
#                         to_insert.append(VehicleTrack(
#                             run_id=val["run_id"],
#                             track_id=val["track_id"],
#                             class_name=val["class_name"],
#                             first_seen=val["first_seen"],
#                             last_seen=val["last_seen"],
#                             metadata=val["metadata"]
#                         ))
#                 if to_insert:
#                     session.bulk_save_objects(to_insert)
#                 if pending_count_events:
#                     session.bulk_save_objects(pending_count_events)
#                 session.commit()
#             except Exception as e:
#                 print("Error committing final pending DB items:", e)

#         cap.release()
#         cv2.destroyAllWindows()
#         end_route_run(session, run)
#         print(f"Finished {route_name}. Totals: {counts}")


# def main():
#     print("Loading model...")
#     # load model and try to put on CUDA if available
#     model = RFDETRBase()
#     try:
#         if torch.cuda.is_available():
#             try:
#                 model = model.to("cuda")
#                 print("Model moved to CUDA")
#             except Exception as e:
#                 # some model wrappers may require different device API; warn but continue
#                 print("Warning: failed to move model to cuda using .to('cuda'):", e)
#         else:
#             print("CUDA not available; running on CPU")
#     except Exception as e:
#         print("Warning checking CUDA availability:", e)

#     tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
#     print("Model + tracker ready.")

#     session = SessionLocal()

#     for rname, vpath in ROUTE_VIDEOS.items():
#         route_obj = get_or_create_route(session, rname, location=vpath, line_config={"type": "horizontal", "y_percent": 0.8})
#         if not Path(vpath).exists():
#             print(f"Warning: {vpath} does not exist. Place the file at this path or update ROUTE_VIDEOS.")
#             continue
#         process_route(rname, vpath, model, tracker, session, route_obj)

#     session.close()
#     print("All routes processed.")


# if __name__ == "__main__":
#     main()
import os
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import torch


from rfdetr import RFDETRBase
from sort.Tracker import Sort

from DB import engine, SessionLocal,get_per_minute_counts,get_hourly_counts,get_class_distribution,get_rolling_5min,Base,Route,RouteRun,VehicleTrack,CountEvent,log_frame_json,export_to_csv


Base.metadata.create_all(bind=engine)

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
INFERENCE_SIZE = (960, 540)
DETECTION_INTERVAL = 2
BATCH_COMMIT_SIZE = 20
SHOW_EVERY_N = 2
LOG_JSON_EVERY = 5
REPORT_INTERVAL = 10

TARGET_CLASSES = ["car", "truck", "bus"]
DISPLAY_MAX_HEIGHT = 900

def safe_show(window_name, frame):
    h, w = frame.shape[:2]
    scale = DISPLAY_MAX_HEIGHT / h
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), DISPLAY_MAX_HEIGHT))
    cv2.imshow(window_name, frame)


def run_analytics(route_id):
    df_min = get_per_minute_counts(route_id)
    df_hour = get_hourly_counts(route_id)
    df_dist = get_class_distribution(route_id)
    df_roll = get_rolling_5min(df_min)

    export_to_csv(df_min, "analytics/per_minute.csv")
    export_to_csv(df_hour, "analytics/hourly.csv")
    export_to_csv(df_dist, "analytics/class_distribution.csv")
    export_to_csv(df_roll, "analytics/rolling_5min.csv")


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


def process_route(route_name, video_path, model, tracker, session, route_obj):
    print(f"Processing {route_name} -> {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    run = start_route_run(session, route_obj, video_path)

    counts = {c: 0 for c in TARGET_CLASSES}
    track_history = {}
    frame_index = 0
    window_name = f"Processing: {route_name}"

    pending_count_events = []
    pending_vehicle_updates = {}
    pending_json_logs = []

    prev_results = None
    class_names = None

    detect_time_ms = 0.0
    track_time_ms = 0.0
    total_time_ms = 0.0

    def scale_boxes_xyxy(boxes, scale_x, scale_y):
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        return boxes

    using_cuda = torch.cuda.is_available()
    model_device = None
    try:
        if hasattr(model, "parameters"):
            p = next(model.parameters(), None)
            if p is not None:
                model_device = p.device
    except Exception:
        model_device = None


    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            loop_start = time.time()
            frame_index += 1
            frame_h, frame_w = frame.shape[:2]
            line_y = int(frame_h * 0.80)

            do_detect = (frame_index % DETECTION_INTERVAL == 0) or (prev_results is None)
            inp_w, inp_h = INFERENCE_SIZE
            small = cv2.resize(frame, (inp_w, inp_h))
            scale_x = frame_w / inp_w
            scale_y = frame_h / inp_h

            results = None
            detect_time_ms = 0.0

            if do_detect:
                detect_start = time.time()
                if using_cuda:
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass

                with torch.inference_mode():
                    results = model.predict(small, threshold=CONFIDENCE_THRESHOLD)

                if using_cuda:
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass

                detect_time_ms = (time.time() - detect_start) * 1000.0
                class_names = results.class_names if hasattr(results, "class_names") else getattr(model, "class_names", None)

                if isinstance(class_names, dict):
                    target_ids = [cid for cid, n in class_names.items() if n in TARGET_CLASSES]
                else:
                    target_ids = [class_names.index(c) for c in TARGET_CLASSES if class_names and c in class_names]

                try:
                    class_id_arr = np.array(results.class_id)
                except Exception:
                    try:
                        class_id_arr = np.array(results.class_id.cpu())
                    except Exception:
                        class_id_arr = np.array([])

                if class_id_arr.size:
                    mask = np.isin(class_id_arr, target_ids)
                    try:
                        filtered = results[mask]
                    except Exception:
                        filtered = results
                else:
                    mask = np.array([], dtype=bool)
                    filtered = results

                xyxy = np.empty((0, 4))
                conf = np.empty((0, 1))
                class_ids = np.empty((0,), dtype=int)

                try:
                    if hasattr(filtered, "xyxy"):
                        xyxy = np.array(filtered.xyxy).astype(float)
                    if hasattr(filtered, "confidence"):
                        conf = np.array(filtered.confidence).astype(float).reshape(-1, 1)
                    if hasattr(filtered, "class_id"):
                        class_ids = np.array(filtered.class_id).astype(int)
                except Exception:
                    pass

                if xyxy.size:
                    xyxy = scale_boxes_xyxy(xyxy.copy(), scale_x, scale_y)

                if xyxy.size:
                    dets = np.hstack((xyxy, conf))
                else:
                    dets = np.empty((0, 5))

                prev_results = {
                    "xyxy": xyxy,
                    "conf": conf,
                    "class_id": class_ids,
                    "dets": dets,
                    "class_names": class_names
                }
            else:
                if prev_results is None:
                    prev_results = {"xyxy": np.empty((0, 4)), "conf": np.empty((0, 1)), "class_id": np.array([]), "dets": np.empty((0, 5)), "class_names": class_names or {}}


            track_start = time.time()
            try:
                tracked_objects = tracker.update(prev_results["dets"])
            except Exception:
                tracked_objects = np.empty((0, 5))
            track_time_ms = (time.time() - track_start) * 1000.0

            detections_log_frame = []
            counts_incremented = {c: 0 for c in TARGET_CLASSES}

            for x1, y1, x2, y2, tid in tracked_objects:
                tid = int(tid)
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                cx = int((x1_i + x2_i) / 2)
                cy = int((y1_i + y2_i) / 2)

                cls_name = "unknown"
                if prev_results["xyxy"].size:
                    try:
                        dists = np.abs(prev_results["xyxy"][:, 0] - x1)
                        idx = int(np.argmin(dists))
                        cls_id = int(prev_results["class_id"][idx]) if prev_results["class_id"].size else None
                        if cls_id is not None:
                            if isinstance(prev_results["class_names"], dict):
                                cls_name = prev_results["class_names"].get(cls_id, "unknown")
                            else:
                                try:
                                    cls_name = prev_results["class_names"][cls_id]
                                except Exception:
                                    cls_name = "unknown"
                    except Exception:
                        cls_name = "unknown"

                prev_y = track_history.get(tid, None)
                crossed = False
                if prev_y is not None and prev_y < line_y <= cy:
                    crossed = True
                    if cls_name in counts:
                        counts[cls_name] += 1
                        counts_incremented[cls_name] += 1
                        pending_count_events.append(
                            CountEvent(run_id=run.id, route_id=route_obj.id, class_name=cls_name, count_increment=1, timestamp=datetime.utcnow())
                        )

                track_history[tid] = cy

                key = (tid, cls_name)
                now = datetime.utcnow()
                existing = pending_vehicle_updates.get(key)
                if existing:
                    existing["last_seen"] = now
                    existing["metadata"] = {"bbox": [x1_i, y1_i, x2_i, y2_i]}
                else:
                    pending_vehicle_updates[key] = {
                        "run_id": run.id,
                        "track_id": tid,
                        "class_name": cls_name,
                        "first_seen": now,
                        "last_seen": now,
                        "metadata": {"bbox": [x1_i, y1_i, x2_i, y2_i]},
                    }

                detections_log_frame.append({
                    "track_id": tid,
                    "class": cls_name,
                    "bbox": [x1_i, y1_i, x2_i, y2_i],
                    "centroid": [cx, cy],
                    "crossed_line": crossed
                })


            if frame_index % LOG_JSON_EVERY == 0:
                try:
                    log_frame_json(route_name, frame_index, detections_log_frame, counts_incremented)
                except Exception:
                    pass
            else:
                pending_json_logs.append((route_name, frame_index, detections_log_frame, counts_incremented))
                if len(pending_json_logs) > 200:
                    for args in pending_json_logs:
                        try:
                            log_frame_json(*args)
                        except Exception:
                            pass
                    pending_json_logs = []

            do_draw = (frame_index % SHOW_EVERY_N == 0)
            if do_draw:
                try:
                    cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 2)

                    for det in detections_log_frame:
                        x1_i, y1_i, x2_i, y2_i = det["bbox"]
                        tid = det["track_id"]
                        cls_name = det["class"]
                        crossed = det["crossed_line"]
                        color = (0, 255, 0) if not crossed else (0, 0, 255)
                        label = f"{cls_name} ID:{tid}"
                        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                        cv2.putText(frame, label, (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    y_offset = 40
                    for v_class, count in counts.items():
                        text = f"{v_class.title()}: {count}"
                        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        y_offset += 40

                    safe_show(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass


            if frame_index % BATCH_COMMIT_SIZE == 0:
                if pending_count_events or pending_vehicle_updates:
                    try:
                        keys = list(pending_vehicle_updates.keys())
                        track_ids = [k[0] for k in keys]
                        existing_tracks = session.query(VehicleTrack).filter(
                            VehicleTrack.run_id == run.id,
                            VehicleTrack.track_id.in_(track_ids)
                        ).all() if track_ids else []

                        existing_map = {(vt.track_id, vt.class_name): vt for vt in existing_tracks}

                        to_insert = []
                        for key, val in pending_vehicle_updates.items():
                            tid, cls = key
                            vt = existing_map.get((tid, cls))
                            if vt:
                                vt.last_seen = val["last_seen"]
                                vt.metadata = val["metadata"]
                            else:
                                to_insert.append(VehicleTrack(
                                    run_id=val["run_id"],
                                    track_id=val["track_id"],
                                    class_name=val["class_name"],
                                    first_seen=val["first_seen"],
                                    last_seen=val["last_seen"],
                                    metadata=val["metadata"]
                                ))

                        if to_insert:
                            session.bulk_save_objects(to_insert)

                        if pending_count_events:
                            session.bulk_save_objects(pending_count_events)

                        session.commit()
                    except Exception as e:
                        print("Error during batch DB commit:", e)
                    finally:
                        pending_count_events = []
                        pending_vehicle_updates = {}

            total_time_ms = (time.time() - loop_start) * 1000.0
            if frame_index % REPORT_INTERVAL == 0:
                fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
                print(f"[{route_name}] Frame {frame_index} | Detect: {detect_time_ms:.2f} ms | Track: {track_time_ms:.2f} ms | Total: {total_time_ms:.2f} ms | FPS: {fps:.2f}")

    finally:
        if pending_json_logs:
            for args in pending_json_logs:
                try:
                    log_frame_json(*args)
                except Exception:
                    pass
            pending_json_logs = []

        if pending_count_events or pending_vehicle_updates:
            try:
                keys = list(pending_vehicle_updates.keys())
                track_ids = [k[0] for k in keys]
                existing_tracks = session.query(VehicleTrack).filter(
                    VehicleTrack.run_id == run.id,
                    VehicleTrack.track_id.in_(track_ids)
                ).all() if track_ids else []

                existing_map = {(vt.track_id, vt.class_name): vt for vt in existing_tracks}
                to_insert = []

                for key, val in pending_vehicle_updates.items():
                    tid, cls = key
                    vt = existing_map.get((tid, cls))
                    if vt:
                        vt.last_seen = val["last_seen"]
                        vt.metadata = val["metadata"]
                    else:
                        to_insert.append(VehicleTrack(
                            run_id=val["run_id"],
                            track_id=val["track_id"],
                            class_name=val["class_name"],
                            first_seen=val["first_seen"],
                            last_seen=val["last_seen"],
                            metadata=val["metadata"]
                        ))

                if to_insert:
                    session.bulk_save_objects(to_insert)
                if pending_count_events:
                    session.bulk_save_objects(pending_count_events)

                session.commit()
            except Exception as e:
                print("Error committing final pending DB items:", e)

        cap.release()
        cv2.destroyAllWindows()
        end_route_run(session, run)
        print(f"Finished {route_name}. Totals: {counts}")


def main():
    print("Loading model...")

    model = RFDETRBase()
    try:
        if torch.cuda.is_available():
            try:
                model = model.to("cuda")
                print("Model moved to CUDA")
            except Exception as e:
                print("Warning: failed to move model to cuda:", e)
        else:
            print("CUDA not available; running on CPU")
    except Exception as e:
        print("Warning checking CUDA availability:", e)

    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
    print("Model + tracker ready.")

    session = SessionLocal()

    for rname, vpath in ROUTE_VIDEOS.items():
        route_obj = get_or_create_route(session, rname, location=vpath, line_config={"type": "horizontal", "y_percent": 0.8})

        if not Path(vpath).exists():
            print(f"Warning: {vpath} does not exist.")
            continue

        process_route(rname, vpath, model, tracker, session, route_obj)

    session.close()
    print("All routes processed.")


if __name__ == "__main__":
    main()

