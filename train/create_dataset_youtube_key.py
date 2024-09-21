import cv2
import numpy as np
import os
from config import CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH
from tensorflow.keras.utils import to_categorical


def preprocess_frames(frames_folder, sequence_length, image_height, image_width):
    frames = []
    frame_files = sorted(os.listdir(frames_folder))[:sequence_length]
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        frames.append(normalized_frame)

    return np.array(frames)


new_frames_folder = "TestFrames"

my_features = []
my_labels_before = []
video_names = []

for class_name in os.listdir(new_frames_folder):
    class_folder = os.path.join(new_frames_folder, class_name)
    if os.path.isdir(class_folder):
        for video_folder in os.listdir(class_folder):
            video_path = os.path.join(class_folder, video_folder)
            frames = preprocess_frames(video_path, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
            if len(frames) == SEQUENCE_LENGTH:
                my_features.append(frames)
                my_labels_before.append(CLASSES.index(class_name))
                video_names.append(video_folder)

my_features = np.array(my_features)
my_labels = to_categorical(my_labels_before, num_classes=len(CLASSES))

np.save("data/key_features.npy", my_features)
np.save("data/key_labels.npy", my_labels)
