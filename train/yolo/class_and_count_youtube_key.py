import cv2
import numpy as np
import os

def load_yolo_model():
    print("Loading YOLO model...")
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    if net.empty():
        print("Failed to load YOLO model")
    return net

def load_class_names():
    print("Loading class names...")
    with open('coco.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    if not class_names:
        print("Failed to load class names")
    return class_names

def detect_objects(frame, net, output_layers):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def process_detections(frame, outs, class_names):
    height, width, channels = frame.shape
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    class_counts = {}
    if len(indices) > 0:
        for i in indices.flatten():
            class_name = class_names[class_ids[i]]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

    return class_counts

# Verify YOLO model and class names loading
net = load_yolo_model()
class_names = load_class_names()
output_layers = get_output_layers(net)

output_file_path = "class_and_count_youtube_key2.txt"
with open(output_file_path, 'w') as f:
    f.write("")

frames_folder = "../Runs/Runs/TestFrames"
if not os.path.exists(frames_folder):
    print(f"Error: The directory {frames_folder} does not exist.")
else:
    for class_name in os.listdir(frames_folder):
        class_folder = os.path.join(frames_folder, class_name)
        print(f"Processing class folder: {class_folder}")
        if os.path.isdir(class_folder):
            for video_folder in os.listdir(class_folder):
                video_path = os.path.join(class_folder, video_folder)
                print(f"Processing video folder: {video_path}")
                if os.path.isdir(video_path):
                    for frame_idx in range(5):
                        frame_file = f"frame_{frame_idx}.jpg"
                        frame_path = os.path.join(video_path, frame_file)
                        if os.path.exists(frame_path):
                            frame = cv2.imread(frame_path)
                            if frame is None:
                                print(f"Error: Unable to read frame {frame_path}.")
                                continue

                            print(f"Frame {frame_idx} read successfully from {frame_path}.")
                            outs = detect_objects(frame, net, output_layers)
                            if not outs:
                                print("No detections were made.")
                            class_counts = process_detections(frame, outs, class_names)
                            print(f"Class counts for {frame_file}: {class_counts}")  # Print class counts

                            result_line = f"{video_folder}.mp4_frame{frame_idx}:" if frame_idx == 0 else f"frame{frame_idx}:"

                            for cls_name, count in class_counts.items():
                                result_line += f"{cls_name}_{count},"

                            result_line = result_line.rstrip(',')
                            result_line += ";"
                            print(f"Result line: {result_line}")

                            with open(output_file_path, 'a') as f:
                                f.write(result_line)
                        else:
                            print(f"Frame {frame_path} does not exist.")
                    with open(output_file_path, 'a') as f:
                        f.write("\n")

print(f"Processing complete. Results are saved in {output_file_path}.")
