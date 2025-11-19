import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase
from sort.Tracker import Sort


def main():
    # =================================================================================
    # STEP 1: INITIAL SETUP
    # =================================================================================
    VIDEO_PATH = r'C:\Users\jaydu\OneDrive\Desktop\python_projects\EVAL\Database\data\test\Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4'
    CONFIDENCE_THRESHOLD = 0.5   # lower threshold improves tracking stability
    
    print("Loading RF-DETR model...")
    model = RFDETRBase()
    print("Model loaded successfully.")

    # SORT tracker
    tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.2)

    # Supervisely annotators (for raw detection)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    print("Initialization complete. Starting video processing...")

    # =================================================================================
    # COUNTERS & TRACK HISTORY
    # =================================================================================
    car_count = 0
    truck_count = 0
    bus_count = 0

    track_history = {}  # track_id → last centroid y

    frame_nmr = 0

    # =================================================================================
    # MAIN LOOP
    # =================================================================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_nmr += 1
        frame_h, frame_w = frame.shape[:2]

        # Counting line (horizontal)
        line_y = int(frame_h * 0.80)

        # =============================================================================
        # STEP 2: DETECTION
        # =============================================================================
        results = model.predict(frame, threshold=CONFIDENCE_THRESHOLD)
        class_names = model.class_names

        target_classes = ["car", "truck", "bus"]

        # RF-DETR gives class_names as {id: name}
        target_ids = [
            cls_id for cls_id, cls_name in class_names.items()
            if cls_name in target_classes
        ]


        mask = np.isin(results.class_id, target_ids)
        filtered = results[mask]

        # =============================================================================
        # STEP 3: PREPARE DETS FOR SORT
        # =============================================================================
        if len(filtered) > 0:
            dets = np.hstack((filtered.xyxy, filtered.confidence.reshape(-1, 1)))
        else:
            dets = np.empty((0, 5))

        # =============================================================================
        # STEP 4: TRACK UPDATE
        # =============================================================================
        tracked_objects = tracker.update(dets)

        # =============================================================================
        # STEP 5: DRAW RAW DETECTIONS
        # =============================================================================
        annotated = frame.copy()

        raw_labels = [
            f"{class_names[c]} {conf:.2f}"
            for c, conf in zip(results.class_id, results.confidence)
        ]

        annotated = box_annotator.annotate(annotated, results)
        annotated = label_annotator.annotate(annotated, results, labels=raw_labels)

        # =============================================================================
        # STEP 6: DRAW TRACKER OUTPUT (GREEN BOXES + ID TEXT)
        # =============================================================================
        for x1, y1, x2, y2, tid in tracked_objects:
            x1, y1, x2, y2, tid = map(int, [x1, y1, x2, y2, tid])

            # draw tracker box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw ID
            cv2.putText(
                annotated,
                f"ID {tid}",
                (x1, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # compute centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # initialize track history
            if tid not in track_history:
                track_history[tid] = cy

            # =============================================================================
            # STEP 7: LINE CROSSING LOGIC
            # =============================================================================
            if track_history[tid] < line_y <= cy:   # crossed downward
                # find the nearest detection to identify class
                distances = np.abs(filtered.xyxy[:, 0] - x1)
                if len(distances) > 0:
                    idx = np.argmin(distances)
                    cls_name = class_names[filtered.class_id[idx]]

                    if cls_name == "car":
                        car_count += 1
                    elif cls_name == "truck":
                        truck_count += 1
                    elif cls_name == "bus":
                        bus_count += 1

            track_history[tid] = cy

        # =============================================================================
        # STEP 8: DRAW COUNTING LINE & TOTAL COUNTS
        # =============================================================================
        cv2.line(annotated, (0, line_y), (frame_w, line_y), (255, 0, 0), 3)

        cv2.putText(annotated, f"CARS: {car_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(annotated, f"TRUCKS: {truck_count}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(annotated, f"BUSES: {bus_count}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # =============================================================================
        # SHOW VIDEO
        # =============================================================================
        # cv2.imshow("Vehicle Detection + Tracking + Counting", annotated)
        cv2.imshow("vehicle Detection + Tracking ",annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # =========================================================================
    # CLEANUP
    # =========================================================================
    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished.")


if __name__ == "__main__":
    main()
