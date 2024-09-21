from config import SEQUENCE_LENGTH, CLASSES, create_dataset
import numpy as np
from tensorflow.keras.utils import to_categorical

trainlist_path = 'txt/cleaned_trainlist01.txt'
testlist_path = 'txt/cleaned_testlist01.txt'

with open(trainlist_path, 'r') as file:
    train_videos = file.readlines()

with open(testlist_path, 'r') as file:
    test_videos = file.readlines()

print(f"Number of training videos: {len(train_videos)}")
print(f"Number of testing videos: {len(test_videos)}")

train_features, train_labels = create_dataset(train_videos, SEQUENCE_LENGTH)
test_features, test_labels = create_dataset(test_videos, SEQUENCE_LENGTH)

print(f"Train features shape: {train_features.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test features shape: {test_features.shape}")
print(f"Test labels shape: {test_labels.shape}")

train_one_hot_encoded_labels = to_categorical(train_labels, num_classes=len(CLASSES))
test_one_hot_encoded_labels = to_categorical(test_labels, num_classes=len(CLASSES))

np.save("data/train.npy", train_features)
np.save("data/train_labels.npy", train_one_hot_encoded_labels)
np.save("data/test.npy", test_features)
np.save("data/test_labels.npy", test_one_hot_encoded_labels)
