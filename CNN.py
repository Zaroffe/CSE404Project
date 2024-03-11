from matplotlib import pyplot as plt

import video_digest

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Create a simple CNN model (replace with your actual model architecture)
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Adjust output layer based on your classes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Generate synthetic data for testing (replace with your actual test data)
num_samples = 100
features_list = []
labels_list = []

for _ in range(num_samples):
    # Generate synthetic features and labels
    depth_map = np.random.random((100, 100)) * 255
    skin_mask = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8) * 255
    hand_size = np.random.randint(5000, 20000)
    label = np.random.randint(0, 2)

    features_list.append([depth_map, skin_mask, hand_size])
    labels_list.append(label)

# Convert the lists to NumPy arrays
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Define CNN model parameters
input_shape = (100, 100, 1)  # Adjust input shape based on your features
num_classes = 2  # Adjust based on the number of gesture classes

# Create and compile the CNN model
cnn_model = create_cnn_model(input_shape)

# Define different experiments with varying features
experiments = [
    {"features": [features_array[:, 0], features_array[:, 1], features_array[:, 2]],
     "labels": labels_array,
     "title": "All Features"},
    # Add more experiments with different combinations of features
]

# Perform experiments and plot the results
for experiment in experiments:
    # Train the model on the experiment's features and labels
    cnn_model.fit(np.expand_dims(experiment["features"][0], axis=-1), experiment["labels"], epochs=10, verbose=0)

    # Evaluate accuracy
    accuracy = video_digest.evaluate_accuracy(experiment["features"], experiment["labels"], cnn_model)
    title = experiment["title"]

    print(f"Accuracy for {title}: {accuracy}")

    # Plot the results (bar chart)
    plt.bar(title, accuracy)

# Show the accuracy comparison
plt.ylabel('Accuracy')
plt.title('Impact of Features on ASL Gesture Recognition Accuracy')
plt.ylim(0, 1)
plt.show()