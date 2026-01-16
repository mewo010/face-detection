# live_detect.py

import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import deque

# ----------------- SETUP -----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

labels = ["smile", "tongue", "mad"]

# Load training data
X = []
y = []

for label in labels:
    data = np.load(f"{label}.npy")
    for landmarks in data:
        # Normalize landmarks for each training sample
        landmarks_np = np.array(landmarks).reshape(-1, 2)
        nose = landmarks_np[1]  # landmark 1 = tip of nose
        landmarks_centered = landmarks_np - nose
        scale = landmarks_np[:,1].max() - landmarks_np[:,1].min()
        if scale != 0:
            landmarks_normalized = (landmarks_centered / scale).flatten()
        else:
            landmarks_normalized = landmarks_centered.flatten()
        X.append(landmarks_normalized)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Prediction smoothing
pred_queues = {}  # store deque for each detected face

# ----------------- VIDEO CAPTURE -----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            # Collect landmarks
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Normalize landmarks relative to nose and face height
            landmarks_np = np.array(landmarks).reshape(-1, 2)
            nose = landmarks_np[1]
            landmarks_centered = landmarks_np - nose
            scale = landmarks_np[:,1].max() - landmarks_np[:,1].min()
            if scale != 0:
                landmarks_normalized = (landmarks_centered / scale).flatten()
            else:
                landmarks_normalized = landmarks_centered.flatten()

            # Predict
            prediction = model.predict([landmarks_normalized])[0]

            # Initialize deque for smoothing
            if i not in pred_queues:
                pred_queues[i] = deque(maxlen=5)
            pred_queues[i].append(prediction)

            # Smoothed prediction
            smooth_pred = max(set(pred_queues[i]), key=pred_queues[i].count)

            # Draw text on frame
            h, w, _ = frame.shape
            x_pos = int(face_landmarks.landmark[0].x * w)
            y_pos = int(face_landmarks.landmark[0].y * h) - 10
            cv2.putText(frame, smooth_pred, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show image
            img = cv2.imread(f"{smooth_pred}.jpg")
            if img is not None:
                img_resized = cv2.resize(img, (200, 200))
                cv2.imshow("Result Image", img_resized)
            else:
                cv2.destroyWindow("Result Image")

    cv2.imshow("Live Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
