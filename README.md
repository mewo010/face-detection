# Face Detection with Expression Recognition

This project allows you to **train a simple facial expression model** using MediaPipe Face Mesh and KNN, and **display images based on the detected expression** in real-time.

---

## Features

- Collect facial landmarks for different expressions using your webcam.
- Train a K-Nearest Neighbors (KNN) classifier with the collected data.
- Real-time face detection and expression recognition.
- Display corresponding images for each detected expression.
- Supports multiple faces and smooth predictions.

---

## Requirements

- Python 3.8+
- OpenCV
- Mediapipe
- NumPy
- scikit-learn

Install dependencies using pip:

```bash
pip install opencv-python mediapipe numpy scikit-learn
