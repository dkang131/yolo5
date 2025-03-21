import cv2
import torch
import numpy as np
from collections import defaultdict
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

model_path = "C:/Users/darren.chee/Desktop/YOLO/yolov5/runs/train/exp/weights/chooch_model.pt"
model = torch.hub.load("C:/Users/darren.chee/Desktop/YOLO/yolov5", 'custom', model_path, source='local')
model.conf = 0.15 # Confidence threshold
model.iou = 0.4  # IoU threshold

def process_video(video_path, model=model, conf=model.conf, iou=model.iou, display_video=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps)

    frame_count = 0
    class_counters = defaultdict(int)

    line_x = 1050
    offset = 35

    tracked_objects = {}
    crossed_objects = set()
    next_object_id = 1

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
            boxes = []
            confidences = []
            class_ids = []

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # Convert to [x, y, w, h]
                confidences.append(float(conf))
                class_ids.append(int(cls))

            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.55)

            # Keep only NMS-filtered detections
            filtered_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confidences[i]
                    cls = class_ids[i]
                    filtered_detections.append([x, y, x + w, y + h, conf, cls])

            # Convert filtered detections to array
            detections = np.array(filtered_detections)

            new_tracked_objects = {}
            assigned_ids = set()

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                class_name = model.names[int(cls)]

                center_x = int(x1 + (x2 - x1) // 2)
                center_y = int(y1 + (y2 - y1) // 2)

                obj_id = None
                min_distance = float("inf")

                for prev_id, (prev_class, prev_x, prev_y, counted) in tracked_objects.items():
                    if prev_class == class_name and prev_id not in assigned_ids:
                        distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            obj_id = prev_id

                if obj_id is None or min_distance > 70:
                    obj_id = next_object_id
                    next_object_id += 1
                assigned_ids.add(obj_id)

                prev_counted = tracked_objects.get(obj_id, (class_name, center_x, center_y, False))[3]
                new_tracked_objects[obj_id] = (class_name, center_x, center_y, prev_counted)

                if obj_id in tracked_objects:
                    prev_class, prev_x, _, counted = tracked_objects[obj_id]
                    # print(f"Tracking ID {obj_id}: {prev_class} moved from {prev_x} to {center_x}")
                    if not counted and prev_x < line_x and center_x >= line_x and obj_id not in crossed_objects:
                        class_counters[prev_class] += 1  # Count only once per object
                        new_tracked_objects[obj_id] = (prev_class, center_x, center_y, True)  # Mark as counted
                        crossed_objects.add(obj_id)  # Store ID to avoid recounting
                        print(f"✅ Counted: Class '{prev_class}' (ID {obj_id}) - Total Count: {class_counters[prev_class]}")
                        results_dict['counts'][prev_class] = class_counters[prev_class]
                    # else:
                        # print(f"❌ Not Counted: ID {obj_id} - prev_x={prev_x}, center_x={center_x}, crossed={obj_id in crossed_objects}")

                # Draw bounding box and confidence label with ID
                if display_video:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame,  f"{class_name} ID:{obj_id} ({int(conf * 100)}%)", 
                                (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            tracked_objects = new_tracked_objects

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
