import numpy as np
import pickle


def load_class_names(names_file_path):
    class_names = {}
    with open(names_file_path, 'r') as file:
        for index, line in enumerate(file):
            class_name = line.strip()
            class_names[class_name] = index
    return class_names


def create_object_vectors_from_file(file_path, class_names, num_classes=80, sequence_length=5):
    # Initialize a dictionary to hold object vectors for each frame
    video_frames_object_vectors = {}
    video_name_count = {}

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                video_name, frames_info = line.split("_frame", 1)

                # Handle duplicate video names
                if video_name in video_name_count:
                    video_name_count[video_name] += 1
                    video_name_unique = f"{video_name}_{video_name_count[video_name]}"
                else:
                    video_name_count[video_name] = 0
                    video_name_unique = video_name

                frames = frames_info.split(";")

                if video_name_unique not in video_frames_object_vectors:
                    video_frames_object_vectors[video_name_unique] = []

                for frame in frames:
                    if frame:
                        frame_data = frame.split(":")
                        if len(frame_data) == 2:
                            _, objects = frame_data
                            object_vector = np.zeros(num_classes)
                            object_list = objects.split(",")
                            for obj in object_list:
                                if "_" in obj:
                                    class_name, count = obj.split("_")
                                    class_index = class_names.get(class_name)
                                    if class_index is not None:
                                        object_vector[class_index] += int(count)
                            video_frames_object_vectors[video_name_unique].append(object_vector)

                # Trim or pad the frames to match the sequence length
                if len(video_frames_object_vectors[video_name_unique]) < sequence_length:
                    pad_length = sequence_length - len(video_frames_object_vectors[video_name_unique])
                    padding = [np.zeros(num_classes) for _ in range(pad_length)]
                    video_frames_object_vectors[video_name_unique].extend(padding)
                else:
                    video_frames_object_vectors[video_name_unique] = video_frames_object_vectors[video_name_unique][
                                                                     :sequence_length]

    return video_frames_object_vectors


def save_object_vectors(file_path, object_vectors):
    with open(file_path, 'wb') as file:
        pickle.dump(object_vectors, file)


def load_object_vectors(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def create_pkl(names_file_path, input_file_path, output_file_path):
    class_names = load_class_names(names_file_path)
    video_frames_object_vectors = create_object_vectors_from_file(input_file_path, class_names)

    # Save the object vectors to a file
    save_object_vectors(output_file_path, video_frames_object_vectors)

    # To load the object vectors from the file
    loaded_object_vectors = load_object_vectors(output_file_path)

    # Convert the dictionary to a numpy array of shape (video_count, sequence_length, num_classes)
    all_videos = np.array(list(loaded_object_vectors.values()))

    # Print the total shape of the loaded object vectors for verification
    print(f'Total shape: {all_videos.shape}')

    # Print each video name and its corresponding shape
    for video_name, frames in loaded_object_vectors.items():
        print(f'Video {video_name}, Shape: {np.array(frames).shape}')


# 'class_and_count_youtube.txt' dosyası için fonksiyonu çağırma
create_pkl('txt/coco.names', 'txt/class_and_count_ucf_test.txt', 'data/ucf_test.pkl')
create_pkl('txt/coco.names', 'txt/class_and_count_ucf_train.txt', 'data/ucf_train.pkl')
create_pkl('txt/coco.names', 'txt/class_and_count_youtube.txt', 'data/youtube.pkl')
create_pkl('txt/coco.names', 'txt/class_and_count_youtube_key2.txt', 'data/youtube_key2.pkl')
