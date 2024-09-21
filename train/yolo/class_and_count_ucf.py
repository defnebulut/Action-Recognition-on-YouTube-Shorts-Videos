import cv2
import numpy as np
from config import SEQUENCE_LENGTH
import os


def load_yolo_model():
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    return net


def load_class_names():
    with open('coco.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
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


def extract_sequence_length_frames(video_path, sequence_length, image_height, image_width):
    cap = cv2.VideoCapture(video_path)
    frames = []
    video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)
    for frame_counter in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = cap.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (image_height, image_width))
        frames.append(resized_frame)
    cap.release()
    return frames


net = load_yolo_model()
class_names = load_class_names()
output_layers = get_output_layers(net)


def create_class_and_count(filename, trainlist_path, videos_folder):
    with open(f"{filename}", 'w') as f:
        f.write("")

    with open(trainlist_path, 'r') as file:
        video_list = [line.strip() for line in file.readlines()]

    for video_entry in video_list:
        class_name, video_file = video_entry.split('/')
        video_path = os.path.join(videos_folder, class_name, video_file)
        frames = extract_sequence_length_frames(video_path, SEQUENCE_LENGTH, 426, 426)

        for idx, frame in enumerate(frames):
            outs = detect_objects(frame, net, output_layers)
            class_counts = process_detections(frame, outs, class_names)

            if idx == 0:
                result_line = f"{video_file}_frame{idx}:"
            else:
                result_line = f"frame{idx}:"

            for class_name, count in class_counts.items():
                result_line += f"{class_name}_{count},"

            result_line = result_line.rstrip(',')
            result_line += ";"
            print(result_line)

            with open(filename, 'a') as f:
                f.write(result_line)
        with open(filename, 'a') as f:
            f.write("\n")


create_class_and_count("class_and_count_ucf_train.txt", "../Runs/Runs/cleaned_trainlist01.txt", "../Runs/Runs/UCF-7")

create_class_and_count("class_and_count_ucf_test.txt", "../Runs/Runs/cleaned_testlist01.txt", "../Runs/Runs/UCF-7")
