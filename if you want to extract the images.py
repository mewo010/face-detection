# test.py

import os
import cv2
import numpy as np

# ----------------- CONFIG -----------------
npy_file = "smile.npy"        # File to extract
output_folder = "test"        # Folder to save images

# ----------------- CREATE FOLDER -----------------
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# ----------------- LOAD .NPY FILE -----------------
data = np.load(npy_file)
print(f"Loaded {npy_file}, shape: {data.shape}")

# ----------------- EXTRACT AND SAVE FRAMES -----------------
for i, frame_landmarks in enumerate(data):
    frame_landmarks = frame_landmarks.reshape(-1, 2)  # 468 landmarks (x, y)

    # Create blank image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw landmarks
    for x, y in frame_landmarks:
        px = int(x * img.shape[1])
        py = int(y * img.shape[0])
        cv2.circle(img, (px, py), 1, (0, 255, 0), -1)

    # Save image
    filename = os.path.join(output_folder, f"frame_{i+1}.png")
    cv2.imwrite(filename, img)

print(f"Saved {len(data)} frames to folder '{output_folder}'")
