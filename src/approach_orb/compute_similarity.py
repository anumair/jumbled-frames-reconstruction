import cv2
import os
import numpy as np
from tqdm import tqdm

def compute_orb_similarity(frames_dir, output_matrix_path="similarity_matrix.npy"):
    # Load all frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]

    print(f"Found {len(frame_paths)} frames.")
    orb = cv2.ORB_create(nfeatures=1000)

    # Step 1: Extract ORB descriptors for all frames
    descriptors = []
    for path in tqdm(frame_paths, desc="Extracting ORB features"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {path}")
            descriptors.append(None)
            continue
        kp, des = orb.detectAndCompute(img, None)
        descriptors.append(des)

    # Step 2: Compute pairwise similarity
    n = len(descriptors)
    similarity_matrix = np.zeros((n, n), dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    print("Computing pairwise similarity matrix...")

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            if descriptors[i] is None or descriptors[j] is None:
                continue
            matches = bf.match(descriptors[i], descriptors[j])
            if len(matches) == 0:
                continue
            # Lower distances mean higher similarity
            distances = [m.distance for m in matches]
            avg_dist = np.mean(distances)
            similarity = 1 / (1 + avg_dist)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Step 3: Save the similarity matrix
    np.save(output_matrix_path, similarity_matrix)
    print(f"Saved similarity matrix to {output_matrix_path}")

    return similarity_matrix


if __name__ == "__main__":
    frames_dir = r"frames"
    output_matrix_path = "src/approach_orb/similarity_matrix.npy"
    compute_orb_similarity(frames_dir, output_matrix_path)
