# Implementation Plan — Retail Theft Detection POC (May 2026 DOCX aligned)

This repo currently implements **multi-route vehicle counting + dashboard** (offline MP4 processing, counting events, SQLite, Streamlit, and an LLM-based “route recommendation” agent). The client DOCX describes a **retail theft detection system** (multi-store/camera ingestion, theft events, alerting, evidence clips, feedback loop, dashboard).

This document proposes a **POC-first implementation plan**, a **proper folder structure**, and **deep logic for each component** so the project can evolve from a single-camera POC to a pilot-ready system.

---

## 1) POC Scope (what to demo)

### POC goal (fast + credible)
Deliver an end-to-end demo on **1 overhead camera feed** (recorded MP4 acceptable) that:
- Detects a theft-like behavior scenario (recommended first: **concealment**).
- Fires an **alert before the person reaches/ crosses an exit line**.
- Produces a **10–20s evidence clip** (with overlay) and a **short explanation**.
- Provides a basic **review UI** (Streamlit) to mark alerts as `confirmed` / `dismissed`.

### Recommended POC scenario
**Concealment**: “item is handled near checkout/bag area and then disappears while person is approaching exit.”

Stretch scenario (Phase 2+):
**Not scanned at checkout**: requires either POS integration, operator-assisted scan markers, audio beep detection, or strong zone-based proxy logic.

---

## 2) Phased Implementation Plan

### Phase 0 — Scope alignment (1–2 days)
**Outputs**
- Decide POC input: recorded MP4 vs RTSP (MP4 recommended first).
- Define 1 camera’s zones in config: `checkout_zone`, `bagging_zone` (optional), `exit_line` / `exit_zone`.
- Define success criteria:
  - Latency target (e.g., < 2–3s processing-to-alert).
  - Noise target (e.g., < 2 false alerts/hour for staged tests).
  - Recall target on staged concealment (e.g., > 80%).

### Phase 1 — Single-camera end-to-end POC (7–14 days)
**Outputs**
- A runnable pipeline: ingest → detect+track → event engine → alert store → evidence clip → dashboard.
- SQLite schema for alerts/events/evidence.
- Evidence clip generation with overlay.
- Deterministic explanations (templates).
- Basic evaluation script for staged clips.

### Phase 2 — Real-time + robustness (2–4 weeks)
**Outputs**
- RTSP ingestion with reconnect, buffering, and backpressure/drop strategy.
- Performance improvements (frame skipping, batching, ONNX/TensorRT option).
- False-positive controls, cooldowns, and operator feedback loop utilization.

### Phase 3 — Multi-camera / multi-store skeleton (2–4 weeks)
**Outputs**
- Multi-store configuration model (stores → cameras).
- Camera health monitoring + centralized alert dashboard.
- Retention policies (hot/warm/cold), audit logs, and basic access control.

---

## 3) Proposed Folder Structure

Create a dedicated package for theft detection. Keep the existing traffic-counting code intact (can be archived later).

```
theft_detection/
  app/
    __init__.py
    main.py                  # CLI entry: run pipeline for camera/video
    config/
      schema.py              # config models + validation
      default.yaml           # sample config with one camera
    ingest/
      video_file.py          # MP4 reader
      rtsp.py                # RTSP reader with reconnect/buffer (Phase 2)
      frame_bus.py           # queue/backpressure abstraction
    vision/
      detector.py            # YOLO wrapper (or other detector)
      tracker.py             # ByteTrack/SORT wrapper
      track_store.py         # track histories + derived signals
    geometry/
      zones.py               # polygons/lines, point-in-poly, crossings
      calibrate.py           # utilities to tune zones (optional)
    events/
      base.py                # EventCandidate dataclasses
      concealment.py         # concealment rule engine (POC)
      not_scanned.py         # scan-mismatch logic (Phase 2+)
      scoring.py             # confidence scoring + thresholds
      suppression.py         # cooldown, de-duplication
    evidence/
      clipper.py             # pre/post ring-buffer clip creation
      overlay.py             # draw tracks/zones overlays
      explanation.py         # deterministic explanation templates
    alerts/
      store.py               # DB write APIs (idempotent)
      notify.py              # webhook/email/sms adapters (Phase 3+)
    db/
      models.py              # SQLAlchemy models
      session.py
    dashboard/
      streamlit_app.py       # alert review UI
      components.py
    evaluation/
      run_eval.py            # evaluate staged videos + labels CSV
      metrics.py
      datasets.py
    utils/
      logging.py
      time.py
      io.py
  scripts/
    init_db.py
    run_demo.ps1
  tests/
    test_zones.py
    test_concealment_rules.py
  README.md
```

### Mapping from the current repo
- Current `routes.py` (video loop + detection + tracking + DB writes) becomes **split** into:
  - `app/ingest/*` (frame source)
  - `app/vision/*` (detector/tracker wrappers)
  - `app/events/*` (theft event rules)
  - `app/alerts/store.py` + `app/db/*` (event/alert persistence)
  - `app/evidence/*` (clip creation + overlay)
- Current `frontend.py` becomes `app/dashboard/streamlit_app.py` but changes domain: **alerts review** instead of traffic analytics.
- Current `DB.py` patterns move into `app/db/models.py` with new entities (`Camera`, `Event`, `Alert`, `Evidence`).

---

## 4) Component Logic (deep dive)

### 4.1 Ingestion (`app/ingest`)
**Objective:** produce timestamped frames reliably, under load.

#### Video file ingestion (POC)
- Iterate frames from MP4.
- Compute timestamps:
  - `ts = start_ts + frame_index / fps` (or use video timestamps if available).
- Emit `FramePacket`:
  - `camera_id`, `frame`, `ts`, `frame_index`.

#### RTSP ingestion (Phase 2)
- Reconnect loop with exponential backoff.
- Use a **bounded queue** (FrameBus) per camera:
  - If queue full: drop oldest (keep “most recent”) or skip (configurable).
- Maintain health status: last frame received time, reconnect count.

#### FrameBus (POC+)
- Bounded queue abstraction:
  - `push(packet)` with drop policy if full.
  - `pop(timeout)` for the inference loop.

Key config knobs:
- `fps_target` (process e.g., 8–15 FPS).
- `max_queue_frames` (e.g., 50–200).
- `drop_policy: oldest|newest|skip`.

---

### 4.2 Vision: detection + tracking (`app/vision`)
**Objective:** stable person-centric tracks + supporting objects.

#### Detector wrapper
- Inputs: frame (BGR), config (img size, thresholds).
- Output: list of detections:
  - `Det { box_xyxy, cls_name, conf }`.
- POC classes:
  - `person`
  - `backpack/handbag` (to support “concealment” reasoning)
  - optional `product` (either generic product class or top SKUs)

#### Tracker wrapper
- Convert detections into tracker format.
- Output stable track IDs:
  - `Track { track_id, cls_name, box_xyxy, conf, last_seen_ts }`
- Maintain track history store:
  - Last N boxes/centroids.
  - Derived signals (speed, direction, time-in-zone).

Performance knobs:
- `detect_every_n_frames` (detector every N frames, tracker runs each frame).
- `img_size`, `conf_thresh`, `iou_thresh`.

---

### 4.3 Geometry & zones (`app/geometry`)
**Objective:** translate pixel coordinates into business intent.

#### Zone definitions per camera
- `checkout_zone`: polygon.
- `bagging_zone`: polygon (optional).
- `exit_line`: line segment (two points) OR `exit_zone` polygon.

#### Geometry utilities
- `point_in_polygon(point, polygon)`
- `crossed_line(prev_point, point, line)` (direction-aware if needed)
- `overlap_ratio(box, polygon)` (for “person overlaps bagging area” signals)

“Alert before exit” gating:
- Compute `distance_to_exit` or “in exit approach zone”.
- Increase event urgency/score as subject gets closer to exit.

---

### 4.4 Event engine: concealment (POC) (`app/events/concealment.py`)
**Objective:** detect “concealment-like” patterns robustly enough for a POC demo.

#### State tracked per person track
- `last_seen_ts`
- `time_in_checkout_zone`
- `near_bag_frames` (count of consecutive frames where person overlaps bagging zone or near detected bag)
- `product_visible_frames` / `product_last_seen_ts` (if product detection is enabled)
- `cooldown_until_ts` (to prevent repeated alerts)

#### Minimal viable concealment rule (configurable thresholds)
Emit `EventCandidate(type="concealment")` when:
1) Subject is in or near `checkout_zone` (or designated high-risk zone).
2) A “product interaction” signal is observed:
   - Option A (preferred): product detected near subject for ≥ `M` frames, then disappears.
   - Option B (fallback): subject enters `bagging_zone` and a “bag interaction” signal spikes (less reliable).
3) Subject overlaps bagging zone or is near `backpack/handbag` for ≥ `K` frames around disappearance time.
4) Subject is approaching exit (within `X` seconds or within `Y` pixels/meters proxy).

False-positive controls:
- Persistence requirement: product must exist for a minimum duration before disappearance.
- De-duplication: only one alert per subject track per `cooldown_seconds`.
- Require multiple signals (zone + interaction + disappearance).

#### Candidate scoring
Compute components:
- `disappearance_duration`
- `bag_overlap_ratio`
- `checkout_relevance`
- `exit_proximity`
Score = weighted sum normalized to [0, 1].

Thresholding:
- If `score >= log_threshold` → store as event for analytics.
- If `score >= alert_threshold` → create alert + evidence clip.

---

### 4.5 Event engine: not-scanned-at-checkout (Phase 2+) (`app/events/not_scanned.py`)
Vision-only “not scanned” is hard without POS/scan signal. Options:
- Operator-assisted scan marker (during staged scenarios).
- Audio beep detection (environment dependent).
- Proxy zones: product enters `bagging_zone` without passing through `scan_zone`.

If implemented as proxy:
- Define `scan_zone` polygon.
- Track product trajectories: `entered_scan_zone` vs `entered_bagging_zone`.
- Candidate if `bagging` occurs without `scan` within `T` seconds.
Mark confidence as lower unless scan signal exists.

---

### 4.6 Evidence generation (`app/evidence`)
**Objective:** produce reviewable proof with context.

#### Clipper (`clipper.py`)
- Maintain a **ring buffer** of frames per camera for `pre_seconds`.
- On event trigger:
  - Freeze last `pre_seconds` frames.
  - Capture next `post_seconds` frames.
  - Write MP4 to `evidence/{camera_id}/{event_id}.mp4`.

#### Overlay (`overlay.py`)
- Draw zones (checkout polygon, exit line).
- Draw subject track bbox + ID.
- Optionally draw product/bag boxes.

#### Explanation (`explanation.py`)
Deterministic template examples:
- “Person track {id} interacted with an item in checkout zone, overlapped bagging area, and the item was not visible for {duration}s while approaching exit.”
Keep deterministic explanation as the source of truth (LLM rewrite is optional later).

---

### 4.7 Alerts & persistence (`app/alerts`, `app/db`)
**Objective:** store events, manage alert lifecycle, capture feedback.

Recommended DB entities (SQLite OK for POC):
- `Camera`: id, store_id, name, source type, url/path, zones JSON, enabled.
- `Event`: id, camera_id, type, subject_track_id, start_ts, end_ts, score JSON, debug JSON.
- `Alert`: id, event_id, status (`open|ack|dismissed|confirmed`), created_ts, resolved_ts, reviewer, feedback.
- `Evidence`: id, event_id, clip_path, overlay_path, thumbnail_path.

Idempotency and suppression:
- `suppression.py` prevents repeated alerts for same subject/event window.
- Use event hashing: `(camera_id, type, subject_track_id, time_bucket)`.

---

### 4.8 Dashboard (`app/dashboard`)
**Objective:** triage and review alerts quickly.

POC screens:
- Alerts list: filters by camera, status, severity, time.
- Alert details: clip player + overlay frame + explanation + action buttons.
- Camera health (basic): last frame time, ingest status.

Feedback loop:
- On `confirmed/dismissed`, write feedback to DB for later retraining and threshold tuning.

---

### 4.9 Evaluation (`app/evaluation`)
**Objective:** quantify POC performance on staged data.

Inputs:
- Staged videos.
- Label CSV: `video, event_type, start_ts, end_ts`.

Metrics:
- Event-level precision/recall (tolerant matching window).
- Average alert latency.
- False alerts per hour per camera.

---

## 5) Minimal Config Shape (conceptual)

The config should be store/camera centric (not “routes”):
- `stores: [{ id, name, cameras: [...] }]`
- `camera: { id, name, source: file|rtsp, path|url, zones: {...} }`
- `vision: { model, img_size, thresholds, detect_every_n }`
- `events: { concealment thresholds, cooldown, log_threshold, alert_threshold }`
- `evidence: { pre_seconds, post_seconds, output_dir }`
- `db: { sqlite_path }`

---

## 6) Practical Next Steps (what to build first)

1) Scaffold folder structure under `theft_detection/app/`.
2) Implement MP4 ingest + detector/tracker wrappers.
3) Implement zone config + geometry helpers.
4) Implement concealment event engine + scoring + suppression.
5) Implement evidence clipper + overlay + deterministic explanation.
6) Implement SQLite models + `alerts/store.py`.
7) Implement Streamlit review UI.
8) Add evaluation script for staged videos.

---

## 7) Decision points to confirm before coding
- POC input: recorded MP4 only, or must include RTSP?
- Primary scenario: concealment only (recommended) vs add not-scanned proxy?
- Which store zone is the “exit” for the camera (line or polygon)?

