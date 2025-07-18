Hand_Gesure

🖱️ Smart Gesture Controller with Voice Feedback This project implements a gesture-controlled virtual mouse using computer vision and hand-tracking, enriched with voice feedback and gesture-based drawing. It uses a webcam to detect hand gestures and perform mouse operations like left-click, right-click, double-click, screenshots, and drawing — with real-time speech responses.

📌 Features ✅ Hand Gesture Detection using MediaPipe 🖱️ Mouse Control: Move cursor using index finger Perform left-click, right-click, double-click 📸 Take Screenshots using a specific gesture 🗣️ Voice Feedback for each action (e.g., "Left click detected") 🎨 Freehand Drawing Mode using gesture 🎤 (Optional) Voice Command Recognition (commented logic in main.py)

🗂️ Project Structure File Description new_main.py Main application file with gesture recognition and action logic util.py Utility functions for angle calculation and distance measurement main.py Prototype file with voice input and speech output support (optional/experimental) my_screenshot_*.png Auto-generated screenshots saved by gesture my_screenshot_88.png, my_screenshot_194.png Example output screenshots

📽️ How It Works Hand Tracking using MediaPipe to detect 21 key landmarks. Gesture Recognition based on angles between fingers and thumb-index distances. Action Execution like clicking or drawing based on detected gesture. Voice Engine (using pyttsx3) announces each action in real time.

🛠️ Requirements Install the required packages: pip install opencv-python mediapipe pyautogui pyttsx3 pynput numpy
