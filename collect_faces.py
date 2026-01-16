# collect_faces.py

import cv2
import mediapipe as mp
import numpy as np

# ----------------- SETUP -----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

cap = cv2.VideoCapture(0)

label = input("Enter face label (example: smile, angry, neutral): ")
data = []

print("Collecting face landmarks... Press 'Q' to stop and save.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Automatically collect all frames
            data.append(landmarks)

    # Show live video
    cv2.imshow("Collect Face", frame)

    # Stop and save with 'Q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save data
data = np.array(data)
np.save(f"{label}.npy", data)
print(f"Saved {len(data)} frames of '{label}' to {label}.npy")
