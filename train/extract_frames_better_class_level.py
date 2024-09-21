import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def process_frame(prev_gray, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_magnitude = np.mean(magnitude)
    return gray, motion_magnitude

def extract_frames_and_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return [], []

    frames = []
    motion_magnitudes = []
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return [], []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(prev_frame)

    frame_batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_batch.append((prev_gray, frame))
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print("No frames extracted from the video.")
        return [], []

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda p: process_frame(*p), frame_batch)

    for result in results:
        _, motion_magnitude = result
        motion_magnitudes.append(motion_magnitude)

    return frames, motion_magnitudes

def extract_feature(frame, model, preprocess_input):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = image.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    feature = model.predict(frame)
    feature = np.squeeze(feature)
    return feature

def extract_features(frames, model, preprocess_input):
    features = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_feature, frame, model, preprocess_input) for frame in frames]
        for future in futures:
            features.append(future.result())
    features_array = np.array(features)  # Convert to numpy array
    print(f"Extracted ResNet features shape: {features_array.shape}")
    return features_array

def color_histogram(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_color_histograms(frames):
    histograms = []
    for frame in frames:
        histograms.append(color_histogram(frame))
    histograms_array = np.array(histograms)
    print(f"Extracted color histograms shape: {histograms_array.shape}")
    return histograms_array

def combine_features(resnet_features, color_histograms):
    combined = np.hstack((resnet_features, color_histograms))
    print(f"Combined features shape: {combined.shape}")
    return combined

def cluster_frames(frames, motion_magnitudes, num_clusters=5):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    resnet_features = extract_features(frames, model, preprocess_input)
    color_histograms = extract_color_histograms(frames)
    combined_features = combine_features(resnet_features, color_histograms)

    if combined_features.size == 0:
        raise ValueError("Combined features array is empty. Check feature extraction steps.")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(combined_features)
    labels = kmeans.labels_

    selected_frames = []
    selected_indices = []
    motion_indices = np.argsort(motion_magnitudes)[::-1]  # Sort indices by motion magnitude in descending order

    for cluster in range(num_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > 0:
            # Select the frame with the highest motion magnitude in the cluster
            motion_cluster_indices = motion_indices[np.isin(motion_indices, cluster_indices)]
            if len(motion_cluster_indices) > 0:
                selected_frame_index = motion_cluster_indices[0]
                selected_frames.append(frames[selected_frame_index])
                selected_indices.append(selected_frame_index)

    # If we have fewer than num_clusters frames, select additional frames from remaining frames
    remaining_indices = [i for i in range(len(frames)) if i not in selected_indices]
    additional_frames_needed = num_clusters - len(selected_frames)
    if additional_frames_needed > 0:
        additional_frames = [frames[i] for i in remaining_indices[:additional_frames_needed]]
        selected_frames.extend(additional_frames)
        selected_indices.extend(remaining_indices[:additional_frames_needed])

    # Sort frames by their original indices to maintain the order
    selected_frames = [frame for _, frame in sorted(zip(selected_indices, selected_frames))]

    return selected_frames, labels


def save_frames(frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"frame_{i}.jpg")  # Adjust extension as needed
        cv2.imwrite(filename, frame)


def process_videos_for_class(class_name, num_samples=5, input_dir='tests2', output_dir='TestFrames'):
    class_dir = os.path.join(input_dir, class_name)

    if not os.path.exists(class_dir):
        print(f"Class directory {class_dir} does not exist.")
        return

    for video_name in os.listdir(class_dir):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(class_dir, video_name)
            video_name_without_ext = os.path.splitext(video_name)[0]

            frames, motion_magnitudes = extract_frames_and_motion(video_path)
            print(
                f"Extracted {len(frames)} frames and {len(motion_magnitudes)} motion magnitudes for {video_name_without_ext}")

            if len(frames) > 0 and len(motion_magnitudes) > 0:
                clustered_frames, _ = cluster_frames(frames, motion_magnitudes, num_samples)

                final_frames = clustered_frames[:num_samples]

                save_frames(final_frames, output_dir=os.path.join(output_dir, class_name, video_name_without_ext))
            else:
                print(f"Error: No frames or motion magnitudes extracted for {video_name_without_ext}.")


process_videos_for_class('Skiing')
