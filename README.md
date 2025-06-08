# Shadow_Fox
Face Expression Detection

# Real-Time Facial Emotion Detection
This project can detect facial emotions in real-time using a camera feed and processes it with OpenCV. The project trains a convolutional neural network (CNN) model on the facial expressions and uses the CNN model to detect emotion from the live webcam video feed. 

# 🔧 Project Structure
1. Training_Model.ipynb
Purpose: To train a CNN model to fit and detect emotional expressions.

Key components:
Loads and pre-processes the emotion dataset.
Creates a CNN using TensorFlow/Keras.
Trains the model and saves both the model architecture (emotiondetector.json) and the weights (emotiondetector.h5).

2. Realtime_Detection.py
Purpose: To run emotion detection in real-time using the webcam.
Key components:
Loads the pre-trained model and weights.
Uses OpenCV Haar Cascade to detect faces.
Detects with the model the emotion from the cropped face.
Displays with bounding boxes and emotion labels the real-time predictions.

# Requirements
Install dependencies using pip:
" pip install numpy opencv-python keras tensorflow "

