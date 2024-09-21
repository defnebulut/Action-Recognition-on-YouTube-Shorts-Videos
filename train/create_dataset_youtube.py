from config import SEQUENCE_LENGTH,IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES, extract_sequence_length_frames
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

videos_folder = "tests2"
my_features = []
my_labels_before = []

for class_name in os.listdir(videos_folder):
    class_folder = os.path.join(videos_folder, class_name)
    if os.path.isdir(class_folder):
        for video_file in os.listdir(class_folder):
            print(video_file)
            video_path = os.path.join(class_folder, video_file)
            frames = extract_sequence_length_frames(video_path, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
            my_features.append(frames)
            my_labels_before.append(CLASSES.index(class_name))

my_features = np.array(my_features)
my_labels = to_categorical(my_labels_before, num_classes=len(CLASSES))

print(f"Features shape: {my_features.shape}")
print(f"Labels shape: {my_labels.shape}")

np.save("data/youtube_features.npy", my_features)
np.save("data/youtube_labels.npy", my_labels)

