# Hand Gesture Volume Control

This project demonstrates a hand gesture recognition system using `OpenCV`, `MediaPipe`, and `PyCaw` libraries. The system controls the system volume based on the distance between two specific points on the user's hand. The application also displays hand landmarks in real-time using the webcam.

## Requirements

Before running the code, ensure you have the following dependencies installed:

1. `opencv-python`
2. `mediapipe`
3. `numpy`
4. `pycaw`

You can install the required libraries using `pip`:

```bash
pip install opencv-python mediapipe numpy pycaw


```


### Notes:
- The first script (`HandDetection.py`) is used to detect and display hand landmarks.
- The second script (`VolumeControl.py`) integrates the hand gesture detection to control system volume using the `pycaw` library.


