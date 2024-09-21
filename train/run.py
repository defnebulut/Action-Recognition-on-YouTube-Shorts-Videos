import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Dense, TimeDistributed, \
    BatchNormalization, Input, concatenate, Layer, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import pickle
from config import SEQUENCE_LENGTH, confusion_matrix_save, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES

RUN = 1
filename = f'exp/Run{RUN}'
os.makedirs(f'{filename}', exist_ok=True)

# Load features and labels
train_features = np.load("./data/train.npy")
train_labels = np.load("./data/train_labels.npy")
test_features = np.load("./data/test.npy")
test_labels = np.load("./data/test_labels.npy")


# Load object vectors
def load_object_vectors(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def map_vector(object_vector):
    mappedVector = []
    for i in range(len(np.array(list(object_vector.values())))):
        mappedVector.append(np.array(list(object_vector.values()))[i])
        mappedVector.append(np.array(list(object_vector.values()))[i])
        mappedVector.append(np.array(list(object_vector.values()))[i])
    return np.array(mappedVector)


# test-train object vectors
train_object_vectors = load_object_vectors('data/ucf_train.pkl')
test_object_vectors = load_object_vectors('data/ucf_test.pkl')

# youtube and youtube-key features and labels
youtube_features = np.load("./data/youtube_features.npy")
youtube_labels = np.load("./data/youtube_labels.npy")
key_features = np.load("./data/key_features.npy")
key_labels = np.load("./data/key_labels.npy")

# youtube and youtube-key vectors
youtube_object_vectors = load_object_vectors('data/youtube.pkl')
youtube_key_object_vectors = load_object_vectors('data/youtube_key2.pkl')

youtube_vector = np.array(np.array(list(youtube_object_vectors.values())))
youtube_key_vector = np.array(np.array(list(youtube_key_object_vectors.values())))
train_vector = map_vector(train_object_vectors)
test_vector = map_vector(test_object_vectors)

# Print the shapes of the object vectors
print(f"train feature shape: {train_features.shape}")
print(f"train vector shape: {train_vector.shape}")
print(f"test feature shape: {test_features.shape}")
print(f"test vector shape: {test_vector.shape}")

print(f"youtube feature shape: {youtube_features.shape}")
print(f"youtube vector shape: {youtube_vector.shape}")
print(f"youtube key feature shape: {key_features.shape}")
print(f"youtube vector shape: {youtube_key_vector.shape}")

# Set random seeds
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Shuffle the training data
train_indices = np.arange(len(train_features))
np.random.shuffle(train_indices)

train_features_shuffled = train_features[train_indices]
train_labels_shuffled = train_labels[train_indices]
train_vector_shuffled = train_vector[train_indices]

# Shuffle the test data
test_indices = np.arange(len(test_features))
np.random.shuffle(test_indices)

test_features_shuffled = test_features[test_indices]
test_labels_shuffled = test_labels[test_indices]
test_vector_shuffled = test_vector[test_indices]

def build_lrcn_model(input_shape_frames, input_shape_objects, num_classes):
    # Frame input
    frames_input = Input(shape=input_shape_frames)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))(frames_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001))(x)

    objects_input = Input(shape=input_shape_objects)

    # Object vector processing with LSTM and additional layers
    y = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(objects_input)
    y = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))(y)
    y = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))(y)

    attention = Attention()([y, y])
    attention = Flatten()(attention)

    combined = concatenate([x, attention])
    combined = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(combined)

    model = Model(inputs=[frames_input, objects_input], outputs=output)
    model.summary()
    return model


input_shape_frames = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
input_shape_objects = (SEQUENCE_LENGTH, len(train_vector[0][0]))
model = build_lrcn_model(input_shape_frames, input_shape_objects, len(CLASSES))

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
callbacks = [early_stopping_callback, reduce_lr_callback]

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

# Train the model
history = model.fit(
    x=[train_features_shuffled, train_vector_shuffled], y=train_labels_shuffled, epochs=100, batch_size=16, shuffle=True,
    validation_data=([test_features_shuffled, test_vector_shuffled], test_labels_shuffled), callbacks=callbacks
)


# Save the model
model.save(f'./{filename}/Model.h5')

# Plot training history
plt.figure(figsize=(14, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f"./{filename}/Accuracy.png")

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"./{filename}/Accuracy_and_Loss.png")

plt.show()

# Save confusion matrix
confusion_matrix_save([youtube_features, youtube_vector], youtube_labels, model, "YouTube", filename)
confusion_matrix_save([key_features, youtube_key_vector], key_labels, model, "YouTubeKeyFrame", filename)
