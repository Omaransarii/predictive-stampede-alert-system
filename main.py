import argparse
import math
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# System Thresholds
SPEED_LIMIT = 20        # Pixel displacement over 5 frames to flag running
CRUSH_RADIUS = 75       # Pixel radius for density clustering
CRUSH_LIMIT = 4         # Minimum neighbors within radius to trigger crush alert
SIDEBAR_WIDTH = 450     # Width of the NumPy telemetry dashboard


def process_video(input_path: str, output_path: str, model_path: str = "yolov8n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width + SIDEBAR_WIDTH, height))

    track_history = defaultdict(list)
    print(f"Processing '{input_path}'...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False
        )
        annotated_frame = results[0].plot(labels=False, conf=False)

        kinematic_alert = False
        global_crush_detected = False
        headcount = 0

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xywh.cpu().tolist()
            headcount = len(boxes)

            current_centroids = [(box[0], box[1]) for box in boxes]

            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x_center, y_center, w, h = box

                # 1. Kinematic Variance (5-frame sliding window)
                track_history[track_id].append((x_center, y_center))
                if len(track_history[track_id]) >= 5:
                    old_x, old_y = track_history[track_id][-5]
                    speed = math.sqrt((x_center - old_x) ** 2 + (y_center - old_y) ** 2)

                    if speed > SPEED_LIMIT:
                        kinematic_alert = True
                        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # 2. Density Clustering (Pairwise Euclidean distance)
                neighbors = sum(
                    1
                    for j, (other_x, other_y) in enumerate(current_centroids)
                    if i != j
                    and math.sqrt((x_center - other_x) ** 2 + (y_center - other_y) ** 2) < CRUSH_RADIUS
                )
                if neighbors >= CRUSH_LIMIT:
                    global_crush_detected = True

        # 3. Matrix Expansion: Build Dual-Screen Command Center Dashboard
        dashboard = np.zeros((height, width + SIDEBAR_WIDTH, 3), dtype=np.uint8)
        dashboard[0:height, 0:width] = annotated_frame
        cv2.rectangle(dashboard, (width, 0), (width + SIDEBAR_WIDTH, height), (30, 30, 30), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        sidebar_x = width + 20

        cv2.putText(dashboard, "AI CONTROL CENTER", (sidebar_x, 50), font, 1.1, (255, 255, 255), 3)
        cv2.line(dashboard, (sidebar_x, 70), (width + SIDEBAR_WIDTH - 20, 70), (200, 200, 200), 2)
        cv2.putText(dashboard, f"LIVE HEADCOUNT: {headcount}", (sidebar_x, 120), font, 0.8, (255, 255, 255), 2)

        threat_text = "NOMINAL"
        threat_color = (0, 255, 0)
        if kinematic_alert and not global_crush_detected:
            threat_text = "ELEVATED (SPEED)"
            threat_color = (0, 255, 255)
        elif global_crush_detected:
            threat_text = "CRITICAL (CRUSH)"
            threat_color = (0, 0, 255)

        cv2.putText(dashboard, "THREAT LEVEL:", (sidebar_x, 190), font, 0.8, (200, 200, 200), 2)
        cv2.putText(dashboard, threat_text, (sidebar_x, 230), font, 1.0, threat_color, 3)

        cv2.putText(dashboard, "SYSTEM LOGS:", (sidebar_x, 320), font, 0.8, (200, 200, 200), 2)
        if threat_text == "NOMINAL":
            cv2.putText(dashboard, "> Crowd flow normal.", (sidebar_x, 360), font, 0.6, (0, 255, 0), 1)
        if kinematic_alert:
            cv2.putText(dashboard, "> Kinematic Variance detected.", (sidebar_x, 390), font, 0.6, (0, 255, 255), 1)
        if global_crush_detected:
            cv2.putText(dashboard, "> DENSITY BOTTLENECK DETECTED!", (sidebar_x, 420), font, 0.6, (0, 0, 255), 2)
            cv2.rectangle(dashboard, (0, 0), (width, height), (0, 0, 255), 15)
            cv2.putText(
                dashboard,
                "WARNING: HIGH DENSITY CRUSH HAZARD",
                (max(20, width // 2 - 250), 60),
                font,
                0.8,
                (255, 255, 255),
                2,
            )

        out.write(dashboard)

    cap.release()
    out.release()
    print(f"Done! Dashboard video saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Stampede Alert System")
    parser.add_argument("--input", type=str, required=True, help="Path to input video (.mp4)")
    parser.add_argument("--output", type=str, default="output_dashboard.mp4", help="Path to output video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 weights")
    args = parser.parse_args()

    process_video(args.input, args.output, args.model)