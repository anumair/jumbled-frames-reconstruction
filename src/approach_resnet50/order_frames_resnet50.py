"""
Frame Ordering using ResNet50 Similarity

This script uses the similarity matrix computed from ResNet50 features
to order frames and reconstruct the video.
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
import time


def greedy_order_frames(similarity_matrix):
    """
    Greedy algorithm to order frames based on similarity.
    Starts from the frame with lowest average similarity (likely first frame)
    and greedily picks the most similar unvisited frame at each step.
    """
    n = similarity_matrix.shape[0]
    visited = [False] * n
    order = []
    
    # Start from frame with lowest average similarity
    avg_sim = similarity_matrix.mean(axis=1)
    start_frame = np.argmin(avg_sim)
    
    print(f"Starting from frame: {start_frame}")
    
    order.append(start_frame)
    visited[start_frame] = True
    
    # Greedily select next frames
    for _ in tqdm(range(n - 1), desc="Ordering frames"):
        current = order[-1]
        similarities = similarity_matrix[current].copy()
        
        # Mask visited frames
        similarities[visited] = -1
        
        # Pick most similar unvisited frame
        next_frame = np.argmax(similarities)
        order.append(next_frame)
        visited[next_frame] = True
    
    return order


def local_refinement_2opt(order, similarity_matrix, max_iterations=100):
    """
    Apply 2-opt local search to improve frame ordering.
    This classic optimization technique tries to reverse segments
    of the path to improve total similarity.
    """
    print("\nApplying 2-opt local refinement...")
    
    n = len(order)
    improved = True
    iteration = 0
    total_improvements = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                # Calculate current cost
                if j == n - 1:
                    current_cost = (similarity_matrix[order[i-1]][order[i]] +
                                  similarity_matrix[order[j-1]][order[j]])
                else:
                    current_cost = (similarity_matrix[order[i-1]][order[i]] +
                                  similarity_matrix[order[j]][order[j+1]])
                
                # Calculate cost after reversing segment [i:j]
                if j == n - 1:
                    new_cost = (similarity_matrix[order[i-1]][order[j-1]] +
                              similarity_matrix[order[i]][order[j]])
                else:
                    new_cost = (similarity_matrix[order[i-1]][order[j-1]] +
                              similarity_matrix[order[i]][order[j+1]])
                
                # If improvement, reverse the segment
                if new_cost > current_cost:
                    order[i:j] = reversed(order[i:j])
                    improved = True
                    total_improvements += 1
    
    print(f"2-opt completed: {total_improvements} improvements in {iteration} iterations")
    return order


def calculate_total_similarity(order, similarity_matrix):
    """
    Calculate the total similarity score of an ordering.
    """
    total = 0
    for i in range(len(order) - 1):
        total += similarity_matrix[order[i]][order[i+1]]
    return total


def save_order_to_file(order, filename="frame_order_resnet50.txt"):
    """Save frame order to text file."""
    with open(filename, "w") as f:
        for idx in order:
            f.write(f"{idx}\n")
    print(f"✅ Frame order saved to: {filename}")


def reconstruct_video(frames_dir, order, output_path, fps=30):
    """
    Reconstruct video from ordered frames.
    """
    print("\nReconstructing video...")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if len(order) != len(frame_files):
        print("⚠️  Warning: Frame count mismatch!")
        return
    
    # Get video properties from first frame
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames in order
    for idx in tqdm(order, desc="Writing video"):
        frame_path = os.path.join(frames_dir, frame_files[idx])
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    
    out.release()
    print(f"✅ Video saved to: {output_path}")


def main():
    """
    Main execution pipeline for ResNet50 approach.
    """
    print("="*60)
    print("ResNet50 FRAME ORDERING & RECONSTRUCTION")
    print("="*60)
    
    # Configuration
    similarity_matrix_path = "src/approach_resnet50/similarity_matrix_resnet50.npy"
    frames_dir = "frames"
    output_order_file = "frame_order_resnet50.txt"
    output_video_path = "output/reconstructed_resnet50.mp4"
    
    start_time = time.time()
    
    # Load similarity matrix
    print("\n📂 Loading similarity matrix...")
    if not os.path.exists(similarity_matrix_path):
        print(f"❌ Error: Similarity matrix not found at {similarity_matrix_path}")
        print("Please run extract_features_resnet50.py first!")
        return
    
    similarity_matrix = np.load(similarity_matrix_path)
    print(f"✅ Loaded similarity matrix: {similarity_matrix.shape}")
    
    # Initial greedy ordering
    print("\n🔍 Running greedy frame ordering...")
    order = greedy_order_frames(similarity_matrix)
    
    # Calculate initial score
    initial_score = calculate_total_similarity(order, similarity_matrix)
    print(f"Initial total similarity: {initial_score:.4f}")
    
    # Apply 2-opt refinement
    order = local_refinement_2opt(order, similarity_matrix, max_iterations=50)
    
    # Calculate final score
    final_score = calculate_total_similarity(order, similarity_matrix)
    print(f"Final total similarity: {final_score:.4f}")
    print(f"Improvement: {final_score - initial_score:.4f}")
    
    # Save order
    print("\n💾 Saving frame order...")
    save_order_to_file(order, output_order_file)
    
    # Reconstruct video
    print("\n🎬 Reconstructing video...")
    reconstruct_video(frames_dir, order, output_video_path)
    
    # Execution summary
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Total frames: {len(order)}")
    print(f"📈 Similarity score: {final_score:.4f}")
    print(f"📁 Output video: {output_video_path}")
    print(f"📄 Frame order: {output_order_file}")
    print("="*60)
    
    # Save execution log
    with open("execution_log_resnet50.txt", "w") as f:
        f.write("ResNet50 Approach Execution Log\n")
        f.write("="*40 + "\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"Total Frames: {len(order)}\n")
        f.write(f"Initial Similarity: {initial_score:.4f}\n")
        f.write(f"Final Similarity: {final_score:.4f}\n")
        f.write(f"Improvement: {final_score - initial_score:.4f}\n")
        f.write(f"Output Video: {output_video_path}\n")
        f.write(f"Frame Order File: {output_order_file}\n")
    
    print("\n✅ Execution log saved to: execution_log_resnet50.txt")


if __name__ == "__main__":
    main()
