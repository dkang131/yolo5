import cv2
import torch
import numpy as np
from collections import defaultdict
import warnings
import sys

sys.path.append("D:/Deepsort_V2/")

from deep_sort.deep_sort import DeepSort

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load YOLO Model
model_path = "C:/Users/darren.chee/Desktop/YOLO/yolov5/runs/train/exp/weights/chooch_model.pt"
model = torch.hub.load("C:/Users/darren.chee/Desktop/YOLO/yolov5", 'custom', model_path, source='local')
model.conf = 0.15  # Confidence threshold
model.iou = 0.4  # IoU threshold

# Initialize DeepSORT
deep_sort = DeepSort("D:/Deepsort_V2/deep_sort/deep/checkpoint/ckpt.t7")

# Store previous positions of tracked objects
previous_positions = {}

def process_video(video_path, model=model, conf=model.conf, iou=model.iou, display_video=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps)

    frame_count = 0
    class_counters = defaultdict(int)

    line_x = 1050
    crossed_objects = set()

    results_dict = {
        'counts': {},
        'processed_frames': 0,
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results_dict['processed_frames'] = frame_count

        if frame_count % 1 == 0:
            results = model(frame)
            detections = results.pred[0].cpu().numpy()

            # Extract bounding boxes, confidences, and class labels
            bbox_xywh = []
            confidences = []
            class_ids = []

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                w, h = x2 - x1, y2 - y1
                bbox_xywh.append([int(x1 + w / 2), int(y1 + h / 2), int(w), int(h)])
                confidences.append(float(conf))
                class_ids.append(int(cls))

            # Convert to numpy arrays
            bbox_xywh = np.array(bbox_xywh, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)

            if bbox_xywh.ndim == 1:
                bbox_xywh = bbox_xywh.reshape(1, -1)

            # Run DeepSORT Tracker
            if len(bbox_xywh) > 0:
                outputs = deep_sort.update(bbox_xywh, confidences, frame)
            else:
                outputs = []

            for output in outputs:
                x1, y1, x2, y2, track_id = output

                center_x = int((x1 + x2) / 2)

                # Assign class name correctly
                if len(class_ids) > 0:
                    class_name = model.names[class_ids[0]]  # Take the first detection class as an approximation
                else:
                    class_name = "Unknown"

                # Check if the object crossed the line
                if track_id in previous_positions:
                    prev_x = previous_positions[track_id]

                    # Ensure it moved **from left to right**
                    if prev_x < line_x and center_x >= line_x and track_id not in crossed_objects:
                        class_counters[class_name] += 1
                        crossed_objects.add(track_id)
                        print(f"✅ Counted: {class_name} (ID {track_id}) - Total Count: {class_counters[class_name]}")

                # Update previous positions
                previous_positions[track_id] = center_x

                # Draw bounding box
                if display_video:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ID:{track_id}", (int(x1), int(y1) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if display_video:
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(frame_delay) & 0xFF == 27:
                break

    results_dict['counts'] = dict(class_counters)
    results_dict['processed_frames'] = frame_count
    cap.release()
    if display_video:
        cv2.destroyAllWindows()

    return results_dict

print(process_video('videos/video_over.mp4', display_video=True))
