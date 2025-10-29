import numpy as np
from tqdm import tqdm

def greedy_order_frames(similarity_matrix, start_frame=None):
    """
    Order frames using greedy nearest neighbor approach.
    
    Args:
        similarity_matrix: NxN similarity matrix
        start_frame: Index of starting frame (None for auto-selection)
    
    Returns:
        List of frame indices in reconstructed order
    """
    n_frames = len(similarity_matrix)
    
    if start_frame is None:
        # Start from the frame with highest total similarity
        start_frame = np.argmax(similarity_matrix.sum(axis=1))
    
    visited = set()
    frame_order = [start_frame]
    visited.add(start_frame)
    
    current = start_frame
    
    for _ in tqdm(range(n_frames - 1), desc="Ordering frames (greedy)"):
        # Get similarities to unvisited frames
        similarities = similarity_matrix[current].copy()
        similarities[list(visited)] = -1  # Mask visited frames
        
        # Select most similar unvisited frame
        next_frame = np.argmax(similarities)
        
        frame_order.append(next_frame)
        visited.add(next_frame)
        current = next_frame
    
    return frame_order

def save_frame_order(frame_order, output_file):
    """Save frame order to text file."""
    with open(output_file, 'w') as f:
        for idx in frame_order:
            f.write(f"{idx}\n")
    print(f"✅ Frame order saved to '{output_file}'")

if __name__ == "__main__":
    similarity_file = "src/approach_resnet_yolo/combined_similarity.npy"
    output_file = "src/approach_resnet_yolo/frame_order.txt"
    
    print("Loading similarity matrix...")
    similarity_matrix = np.load(similarity_file)
    
    print("Computing optimal frame order...")
    frame_order = greedy_order_frames(similarity_matrix)
    
    save_frame_order(frame_order, output_file)
    print(f"Reconstructed sequence length: {len(frame_order)}")
