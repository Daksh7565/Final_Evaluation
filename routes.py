from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import torch
from rfdetr import RFDETRBase
from sort.Tracker import Sort
from DB import (
    engine,
    SessionLocal,
    Base,
    Route,
    RouteRun,
    VehicleTrack,
    CountEvent,
    log_frame_json,
)
import yaml
def load_config(config_path="config.yaml"):
    try:
        with open(config_path,'r',encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        print("error: file not found")
config=load_config()

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

# ------------------
# Configuration
# ------------------
ROUTE_VIDEOS = {
    "route_01": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4",
    "route_02": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Cars Moving On Road Stock Footage - Free Download.mp4",
    "route_03": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\videoplayback (online-video-cutter.com).mp4",
    "route_04": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\27260-362770008_large.mp4",
    "route_05": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-534232220-640_adpp_is.mp4",
    "route_06": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-866517852-640_adpp_is.mp4",
    "route_07": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\istockphoto-1282097063-640_adpp_is.mp4",
    "route_08": r"C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\2103099-uhd_3840_2160_30fps (1).mp4",
}

ROUTE_LOCATIONS = {
    "route_01": "Highway Camera A - Northbound",
    "route_02": "City Road East - Midblock",
    "route_03": "Intersection 7 - South",
    "route_04": "Bypass Road - Near Junction",
    "route_05": "Suburban Road - West",
    "route_06": "Highway Sector 12",
    "route_07": "Tunnel Exit Camera",
    "route_08": "Market Road Central",
}


CONFIDENCE_THRESHOLD = config["video_processing"]["detection"]["confidence_threshold"]
INFERENCE_SIZE = config["video_processing"]["detection"]["inference_size"]
DETECTION_INTERVAL = config["video_processing"]["performance"]["detection_interval"]
BATCH_COMMIT_SIZE = config["video_processing"]["performance"]["batch_commit_size"]
SHOW_EVERY_N = config["video_processing"]["visualization"]["show_every_n"]
LOG_JSON_EVERY = 5
REPORT_INTERVAL = 100  
TARGET_CLASSES = config["video_processing"]["detection"]["target_classes"]
DISPLAY_MAX_HEIGHT = config["video_processing"]["visualization"]["display_max_height"]

SNAPSHOT_DIR = Path("processed_frames")
SNAPSHOT_DIR.mkdir(exist_ok=True)  
SNAPSHOT_INTERVAL = 30  


# ------------------
# Helpers
# ------------------

def safe_show(window_name: str, frame: np.ndarray):
    """Show a frame scaled down if too tall to fit screen."""
    h, w = frame.shape[:2]
    scale = DISPLAY_MAX_HEIGHT / h
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), DISPLAY_MAX_HEIGHT))
    cv2.imshow(window_name, frame)


def get_or_create_route(session, name: str, location: str = None, line_config: dict = None):
    """Fetch route by name or create with a sane default location."""
    route = session.query(Route).filter_by(name=name).first()
    if route:
        if line_config and (route.line_config is None):
            route.line_config = line_config
            session.commit()
            session.refresh(route)
        return route

    route = Route(name=name, location=location, line_config=line_config)
    session.add(route)
    session.commit()
    session.refresh(route)
    return route


def start_route_run(session, route: Route, video_path: str) -> RouteRun:
    run = RouteRun(route_id=route.id, raw_video_path=video_path, start_time=datetime.utcnow())
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def end_route_run(session, run: RouteRun):
    run.end_time = datetime.utcnow()
    session.commit()


# ------------------
# Main processing function
# ------------------

def process_route(route_name: str, video_path: str, model, tracker, session, route_obj: Route):
    """Process a single route video: detect, track, count, log JSON and write DB events in batches."""
    print(f"Processing {route_name} -> {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    run = start_route_run(session, route_obj, video_path)

    counts = {c: 0 for c in TARGET_CLASSES}
    track_history = {}
    active_tracks = {} 
    frame_index = 0
    window_name = f"Processing: {route_name}"

    pending_count_events = []
    prev_results, class_names = None, None

    def scale_boxes_xyxy(boxes: np.ndarray, scale_x: float, scale_y: float):
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        return boxes

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]

            line_percent = 0.80
            if getattr(route_obj, "line_config", None):
                try: line_percent = float(route_obj.line_config.get("y_percent", line_percent))
                except (ValueError, TypeError): line_percent = 0.80
            line_y = int(frame_h * line_percent)

            do_detect = (frame_index % DETECTION_INTERVAL == 0) or (prev_results is None)
            
            if do_detect:
                inp_w, inp_h = INFERENCE_SIZE[0],INFERENCE_SIZE[1]
                small = cv2.resize(frame, (inp_w, inp_h))
                with torch.inference_mode():
                    results = model.predict(small, threshold=CONFIDENCE_THRESHOLD)
                
                class_names = getattr(results, "class_names", getattr(model, "class_names", None))
                target_ids = [k for k, v in (class_names or {}).items() if v in TARGET_CLASSES]
                
                class_id_arr = np.array(results.class_id.cpu() if hasattr(results.class_id, 'cpu') else results.class_id, dtype=int)
                xyxy = np.array(results.xyxy, dtype=float)
                conf = np.array(results.confidence, dtype=float).reshape(-1, 1)

                mask = np.isin(class_id_arr, target_ids)
                filtered_xyxy = scale_boxes_xyxy(xyxy[mask].copy(), frame_w / inp_w, frame_h / inp_h)
                filtered_conf = conf[mask]
                filtered_class_ids = class_id_arr[mask]
                
                dets = np.hstack((filtered_xyxy, filtered_conf)) if filtered_xyxy.size else np.empty((0, 5))
                prev_results = {"xyxy": filtered_xyxy, "conf": filtered_conf, "class_id": filtered_class_ids, "dets": dets, "class_names": class_names}
            
            tracked_objects = tracker.update(prev_results["dets"]) if prev_results else np.empty((0, 5))
            
            detections_log_frame = []
            counts_incremented = {c: 0 for c in TARGET_CLASSES}
            
            # Get current timestamp once per frame for consistency
            current_time = datetime.utcnow()

            for x1, y1, x2, y2, tid in tracked_objects:
                tid = int(tid)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                cls_name = "unknown"
                if prev_results["xyxy"].size > 0:
                    # Associate detection with the closest tracked object
                    dists = np.linalg.norm(prev_results["xyxy"][:, :2] - np.array([x1, y1]), axis=1)
                    idx = np.argmin(dists)
                    cls_id = int(prev_results["class_id"][idx])
                    cls_name = prev_results["class_names"].get(cls_id, "unknown")

                if tid not in active_tracks:
                    # First time seeing this track ID, create a new record
                    new_track = VehicleTrack(
                        run_id=run.id,
                        track_id=tid,
                        class_name=cls_name,
                        first_seen=current_time,
                        last_seen=current_time
                    )
                    session.add(new_track)
                    active_tracks[tid] = new_track
                else:
                    active_tracks[tid].last_seen = current_time
                
                prev_y = track_history.get(tid)
                crossed = False
                if prev_y is not None and prev_y < line_y <= cy:
                    crossed = True
                    if cls_name in counts:
                        counts[cls_name] += 1
                        counts_incremented[cls_name] += 1
                        pending_count_events.append(CountEvent(run_id=run.id, route_id=route_obj.id, class_name=cls_name, count_increment=1, timestamp=current_time))
                
                track_history[tid] = cy
                
                detections_log_frame.append({"track_id": tid, "class": cls_name, "bbox": [int(x1), int(y1), int(x2), int(y2)], "centroid": [cx, cy], "crossed_line": crossed})

            if frame_index % LOG_JSON_EVERY == 0 and detections_log_frame:
                log_frame_json(route_name, frame_index, detections_log_frame, counts_incremented)

            # --- Visualization Block ---
            cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 255, 255), 2)
            for det in detections_log_frame:
                x1_i, y1_i, x2_i, y2_i = det["bbox"]
                label = f"{det['class']} ID:{det['track_id']}"
                color = (0, 0, 255) if det["crossed_line"] else (0, 255, 0)
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                cv2.putText(frame, label, (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            y_offset = 50
            padding = 20
            font_scale = 1.2
            font_thickness = 2
            for v_class, count in sorted(counts.items()):
                text = f"{v_class.title()}: {count}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = padding
                text_y = y_offset
                cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), (text_x + text_width + 10, text_y + 10), (0,0,0), -1)
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                y_offset += text_height + padding + 10

            if frame_index % SHOW_EVERY_N == 0:
                safe_show(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break

            if frame_index % SNAPSHOT_INTERVAL == 0:
                snapshot_path = SNAPSHOT_DIR / f"{route_name}_latest.jpg"
                cv2.imwrite(str(snapshot_path), frame)

            if frame_index % BATCH_COMMIT_SIZE == 0:
                if pending_count_events:
                    session.bulk_save_objects(pending_count_events)
                    pending_count_events = []
                session.commit()

            if frame_index % REPORT_INTERVAL == 0:
                print(f"[{route_name}] Frame {frame_index} | Totals: {counts}")

    finally:
        if pending_count_events:
            session.bulk_save_objects(pending_count_events)
        session.commit()
        
        cap.release()
        cv2.destroyAllWindows()
        end_route_run(session, run)
        print(f"Finished {route_name}. Totals: {counts}")

def main():
    print("Loading model...")
    model = RFDETRBase()
    try:
        if torch.cuda.is_available(): model = model.to("cuda"); print("Model moved to CUDA")
        else: print("CUDA not available; running on CPU")
    except Exception as e:
        print(f"Warning checking CUDA availability: {e}")

    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)
    print("Model + tracker ready.")

    with SessionLocal() as session:
        for rname, vpath in ROUTE_VIDEOS.items():
            if not Path(vpath).exists():
                print(f"Warning: {vpath} does not exist. Skipping route {rname}.")
                continue
            
            location = ROUTE_LOCATIONS.get(rname, f"Simulated camera for {rname}")
            line_conf = {"type": "horizontal", "y_percent": 0.8}
            route_obj = get_or_create_route(session, rname, location=location, line_config=line_conf)
            process_route(rname, vpath, model, tracker, session, route_obj)

    print("All routes processed.")

if __name__ == "__main__":
    main()