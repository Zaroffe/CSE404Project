#this should all work, unless I broke it last night and completely forgot about it.
#making it into a git repository may have screwed with the packages as well.

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to extract features from a frame
def extract_features(frame):
    # Implement the feature extraction methods as before
    depth_map = frame
    skin_mask = extract_skin_color(frame)
    hand_size = extract_hand_size(skin_mask)
    return depth_map, skin_mask, hand_size

# Function to evaluate the accuracy of ASL gesture recognition
def evaluate_accuracy(features, labels, model):
    predictions = []

    for feature_set in features:
        # Process the features and predict the gesture class using your CNN model
        gesture_class = predict_gesture(feature_set, model)  # You need to implement this function
        predictions.append(gesture_class)

    accuracy = accuracy_score(labels, predictions)
    return accuracy

# Function to predict gesture using your CNN model
def predict_gesture(feature_set, model):
    # Implement the prediction using your CNN model
    # For simplicity, we'll assume a binary classification
    # Replace this with the actual prediction logic for your model
    predicted_class = model.predict(np.array([feature_set]))[0]
    gesture_class = np.argmax(predicted_class)
    return gesture_class

# Function to extract skin color using color segmentation
def extract_skin_color(frame):
    # Replace this with the actual skin color extraction logic
    # The example below assumes a binary mask, replace with your method
    skin_mask = np.random.randint(0, 2, size=frame.shape[:2], dtype=np.uint8) * 255
    return skin_mask

# Function to extract hand size using contour detection
def extract_hand_size(skin_mask):
    # Replace this with the actual hand size extraction logic
    # The example below assumes a random hand size, replace with your method
    hand_size = np.random.randint(5000, 20000)
    return hand_size


def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)

        # Ensure each frame is correctly preprocessed
        if preprocessed_frame is None or preprocessed_frame.size == 0:
            print("Preprocessing failed or returned empty frame.")
            continue

        frames.append(preprocessed_frame)

    cap.release()

    frames = np.array(frames)

    # Check if the frames array is not empty
    if frames.size == 0:
        print("No frames to predict on.")
        return None

    # Log the shape to ensure it's correct
    print("Frames shape:", frames.shape)

    try:
        predictions = model.predict(frames)
    except ValueError as e:
        print(f"Error during model prediction: {e}")
        # Optionally, add more diagnostic information here
        return None

    # Postprocess predictions if necessary
    result = postprocess_predictions(predictions)
    return result
# Function to preprocess a frame (resize, normalize, etc.)
def preprocess_frame(frame):
    # Implement frame preprocessing steps here
    # For example, resize the frame to the input size expected by your model
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values if needed
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Function to postprocess predictions (e.g., find the most common prediction)
def postprocess_predictions(predictions):
    # Implement postprocessing steps here
    # For example, find the class with the highest average prediction across frames
    average_predictions = np.mean(predictions, axis=0)
    result = np.argmax(average_predictions)
    return result