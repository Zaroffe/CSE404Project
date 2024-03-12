#this is where we will assemble all the functions
#the order should be: break up video, create features, train CNN, run CNN, convert broken english to proper english

import CNN
import video_digest
import proper_englishifier

model = CNN.create_cnn_model()
# Load your trained CNN model
# trained_model = ...  # Load your model using TensorFlow or Keras

# Example usage:
# video_path = "path/to/your/test_gesture_video.mp4"     #change to your video path
# result = video_digest.process_video(video_path, trained_model)
# print("Predicted gesture:", result)