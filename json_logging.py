# json_logger.py
import json
from pathlib import Path
from datetime import datetime, timezone

LOG_DIR = Path("logs/json")
LOG_DIR.mkdir(parents=True, exist_ok=True)

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
