import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import video_digest

def create_cnn_model(input_shape):
    num_classes = 3  # Adjust this based on the number of classes
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Replace these with your actual training data and labels
# IE: send training data through video digest, and assign to X and y
X_train = np.random.rand(100, 224, 224, 3)  # Example feature data
y_train = np.random.randint(0, 2, size=(100,))  # Example labels

input_shape = X_train.shape[1:]  # Determine input shape dynamically

cnn_model = create_cnn_model(input_shape)

# Train model (ensure you have actual data for this)
cnn_model.fit(X_train, y_train, epochs=10, verbose=1)

video_path = "path/to/your/video.mp4" #CHANGE ME
predicted_result = video_digest.process_video(video_path, cnn_model)

print("Predicted gesture class for the video:", predicted_result)
