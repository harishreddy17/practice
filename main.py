import cv2
import numpy as np
import time
from collections import defaultdict
import torch
import os
import pandas as pd
from utils_custom.interactions import check_interaction
from utils_custom.Object_Count import update_object_counts
from utils_custom.Object_Count import display_counts


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

object_classes = ['person', 'chair', 'bottle', 'laptop', 'car']

cap = cv2.VideoCapture('Test Videos\demo.mp4')


output_path = 'output_video.mp4' 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

annotation_folder = 'annots/obj_train_data'
  

frame_object_counts = defaultdict(int)

frame_count = 0
fps = 0
start_time = time.time()
total_tp, total_fp, total_fn = 0, 0, 0


def detect_objects(frame):
    results = model(frame)
    results_df = results.pandas().xyxy[0]
    return results_df[results_df['name'].isin(object_classes)]


def load_ground_truth(frame_number):
    """Load ground truth data from the annotation file corresponding to the frame number."""
    annotation_file = os.path.join(annotation_folder, f"frame_{frame_number:06d}.txt") 
    print(f"Loading annotation file: {annotation_file}")
    ground_truth = []
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as file:
            for line in file:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                ground_truth.append((int(class_id), x_center, y_center, width, height))
    return ground_truth


def calculate_iou(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_detection(predictions, ground_truth, frame):
    """Evaluate detection accuracy using IoU between predictions and ground truth."""
    global total_tp, total_fp, total_fn
    iou_threshold = 0.3 
    tp, fp, fn = 0, 0, 0

    print(f"Evaluating frame {frame_count} with dimensions {frame.shape}...")
    
    for _, pred in predictions.iterrows():
        pred_box = (pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'])
        pred_class = object_classes.index(pred['name']) if pred['name'] in object_classes else -1

        matched = False
        for gt_class, x_center, y_center, width, height in ground_truth:

            gt_box = (
                (x_center - width / 2) * frame.shape[1],
                (y_center - height / 2) * frame.shape[0],
                (x_center + width / 2) * frame.shape[1],
                (y_center + height / 2) * frame.shape[0]
            )

            iou = calculate_iou(pred_box, gt_box)
            print(f"Pred: {pred['name']} at {pred_box}, GT: Class {gt_class} at {gt_box}, IoU: {iou}")

            if iou >= iou_threshold and pred_class == gt_class:
                tp += 1
                matched = True
                print("Match found!")
                break
        if not matched:
            fp += 1
        

    fn = len(ground_truth) - tp
    total_tp += tp
    total_fp += fp
    total_fn += fn


def draw_results(frame, results, interactions):
    for _, row in results.iterrows():
        if row['name'] in object_classes: 
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}" 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for obj1, obj2, interaction in interactions:
        text_size = cv2.getTextSize(interaction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = 350
        text_y = 30
        text_bg_x2 = text_x + text_size[0] + 10  # Extra padding for the background
        text_bg_y2 = text_y + text_size[1] + 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), (text_bg_x2, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, interaction, (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



def display_fps(frame):
    global frame_count, start_time
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 1 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame


def main():
    global frame_count
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        results = detect_objects(frame)
        update_object_counts(results)
        interactions = check_interaction(results)
        draw_results(frame, results, interactions)
        display_counts(frame)
        frame = display_fps(frame)

    
        ground_truth = load_ground_truth(frame_count)

        evaluate_detection(results, ground_truth, frame)

        out.write(frame)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()