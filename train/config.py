import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score

DATASET_DIR = "UCF-8"
CLASSES = ["ApplyEyeMakeup", "BasketballDunk", "Diving", "IceDancing", "HorseRiding", "PlayingGuitar",
           "BlowingCandles", "WalkingWithDog"]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 5


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
        normalized_frame = resized_frame / 255
        frames.append(normalized_frame)
    cap.release()
    return frames


def zoom_augmentation(frames, zoom_factor=1.2):
    augmented_frames = []
    for frame in frames:
        height, width, channels = frame.shape
        new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)
        top, bottom = (height - new_height) // 2, (height + new_height) // 2
        left, right = (width - new_width) // 2, (width + new_width) // 2
        cropped_frame = frame[top:bottom, left:right]
        zoomed_frame = cv2.resize(cropped_frame, (width, height))
        augmented_frames.append(zoomed_frame)
    return augmented_frames


def augment_frames(frames):
    augmented_frames = [frames]
    mirrored_frames = [cv2.flip(frame, 1) for frame in frames]
    zoomed_frames = zoom_augmentation(frames)
    augmented_frames.append(mirrored_frames)
    augmented_frames.append(zoomed_frames)
    return augmented_frames


def create_dataset(video_folder, sequence_length):
    features = []
    labels = []
    for video in video_folder:
        video_path = os.path.join(DATASET_DIR, video.strip())
        class_name = video.split('/')[0]
        if class_name in CLASSES:
            class_index = CLASSES.index(class_name)
            frames = extract_sequence_length_frames(video_path, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
            if len(frames) == sequence_length:
                augmented_frames = augment_frames(frames)
                for aug_frames in augmented_frames:
                    features.append(aug_frames)
                    labels.append(class_index)
    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels


def confusion_matrix_save(test_features, test_labels, model, dataset_origin, filename):
    predictions = model.predict(test_features)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    my_loss, my_accuracy = model.evaluate(test_features, test_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    report = classification_report(true_labels, predicted_labels, target_names=CLASSES)

    with open(f"./{filename}/{dataset_origin}_metrics.txt", 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(f"./{filename}/{dataset_origin}_ConfusionMatrix_Accuracy_{my_accuracy}_Loss_{my_loss}.png")
    plt.show()

