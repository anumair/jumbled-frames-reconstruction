import numpy as np
import cv2
import os

def heuristic_ordering(similarity_matrix, alpha=0.8, beta=0.2, k=5):
    """
    Heuristic Greedy Frame Ordering
    --------------------------------
    - Treats frame ordering as a path reconstruction problem similar to TSP.
    - Maximizes similarity between consecutive frames using greedy + heuristic rules.
    """
    n = similarity_matrix.shape[0]
    used = set()
    order = []

    # Start from the frame with the highest average similarity
    avg_sim = similarity_matrix.mean(axis=1)
    current = np.argmin(avg_sim)
    order.append(current)
    used.add(current)

    print(f"Starting frame: {current}")

    while len(order) < n:
        sims = similarity_matrix[current].copy()
        sims[list(used)] = -1  # ignore already used frames

        # Pick top-k most similar unused frames
        candidate_indices = np.argsort(sims)[-k:]
        best_score = -np.inf
        next_frame = None

        for j in candidate_indices:
            # Smoothness heuristic using last few frames
            recent_frames = order[-3:] if len(order) >= 3 else order
            smoothness = np.mean([similarity_matrix[j][r] for r in recent_frames])
            score = alpha * sims[j] + beta * smoothness
            if score > best_score:
                best_score = score
                next_frame = j

        order.append(next_frame)
        used.add(next_frame)
        current = next_frame

        if len(order) % 50 == 0 or len(order) == n:
            print(f"Progress: {len(order)}/{n} frames ordered...")

    return order


def local_refinement(order, similarity_matrix, window_size=10):
    """
    Local refinement to fix small misplaced clusters
    by checking short sub-sequences and swapping frames
    if it improves local similarity continuity.
    """
    n = len(order)
    improved = 0

    for i in range(0, n - window_size, window_size // 2):
        segment = order[i:i + window_size]
        best_segment = segment.copy()
        best_score = segment_score(segment, similarity_matrix)

        # Try simple pairwise swaps in the segment
        for a in range(len(segment)):
            for b in range(a + 1, len(segment)):
                swapped = segment.copy()
                swapped[a], swapped[b] = swapped[b], swapped[a]
                score = segment_score(swapped, similarity_matrix)
                if score > best_score:
                    best_score = score
                    best_segment = swapped
                    improved += 1

        order[i:i + window_size] = best_segment

    print(f"Local refinement completed with {improved} improvements.")
    return order


def segment_score(segment, similarity_matrix):
    """Calculate total similarity score of consecutive frames in a segment."""
    score = 0
    for i in range(len(segment) - 1):
        score += similarity_matrix[segment[i]][segment[i + 1]]
    return score


def reconstruct_video(frames_dir, frame_order, output_video_path, fps=30):
    """
    Stage 4: Video Reconstruction
    - Reads frames in the predicted order.
    - Writes final video using OpenCV VideoWriter.
    """
    frame_files = sorted(os.listdir(frames_dir))
    if not frame_files:
        print("Error: No frames found in the frames directory.")
        return

    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i, idx in enumerate(frame_order):
        frame_path = os.path.join(frames_dir, frame_files[idx])
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"Writing frame {i + 1}/{len(frame_order)}")

    out.release()
    print(f"Reconstructed video saved at: {output_video_path}")


if __name__ == "__main__":
    # File paths
    similarity_matrix_path = r"src\approach_orb\similarity_matrix.npy"
    frames_dir = r"frames"
    output_order_path = r"src\approach_orb\frame_order_heuristic.txt"
    output_video_path = r"output\reconstructed_orb.mp4"

    # Stage 3: Frame ordering
    if not os.path.exists(similarity_matrix_path):
        print("Error: Similarity matrix not found at", similarity_matrix_path)
        exit()

    print("Loading similarity matrix...")
    sim_matrix = np.load(similarity_matrix_path)
    print("Running heuristic-based greedy frame ordering...")
    frame_order = heuristic_ordering(sim_matrix)

    # Optional refinement
    print("Refining locally for smoother transitions...")
    frame_order = local_refinement(frame_order, sim_matrix)

    # Save order
    with open(output_order_path, "w") as f:
        for idx in frame_order:
            f.write(str(idx) + "\n")

    print(f"Frame ordering saved to {output_order_path}")

    # Stage 4: Video reconstruction
    print("Reconstructing video...")
    reconstruct_video(frames_dir, frame_order, output_video_path)
