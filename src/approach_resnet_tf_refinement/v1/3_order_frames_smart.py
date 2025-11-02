import numpy as np
from tqdm import tqdm

def find_optimal_start_point(similarity_matrix):
    """
    Find the optimal starting point by detecting the beginning of the sequence.
    Uses multiple heuristics to identify the start frame.
    
    Args:
        similarity_matrix: NxN similarity matrix
    
    Returns:
        Index of optimal starting frame
    """
    n_frames = len(similarity_matrix)
    
    print("\n=== Finding Optimal Start Point ===")
    
    # Heuristic 1: Find frame with lowest average similarity to future frames
    # (Start frames are less similar to middle/end frames)
    avg_similarities = similarity_matrix.mean(axis=1)
    candidate1 = np.argmin(avg_similarities)
    print(f"Heuristic 1 (lowest avg similarity): Frame {candidate1}")
    
    # Heuristic 2: Find frame that starts a long high-similarity chain
    # Build greedy path from each frame and see which gives longest initial chain
    best_start = 0
    best_chain_length = 0
    
    print("Testing start points...")
    for start_idx in tqdm(range(n_frames), desc="Evaluating starts"):
        visited = set([start_idx])
        current = start_idx
        chain_similarity = 0
        
        # Build short chain (first 50 frames)
        for _ in range(min(50, n_frames-1)):
            similarities = similarity_matrix[current].copy()
            similarities[list(visited)] = -1
            
            next_frame = np.argmax(similarities)
            if similarities[next_frame] < 0.85:  # Similarity threshold
                break
                
            chain_similarity += similarities[next_frame]
            visited.add(next_frame)
            current = next_frame
        
        if chain_similarity > best_chain_length:
            best_chain_length = chain_similarity
            best_start = start_idx
    
    print(f"Heuristic 2 (best chain start): Frame {best_start} (chain score: {best_chain_length:.2f})")
    
    # Heuristic 3: Find endpoint and reverse
    # Endpoint has high similarity to one side but low to the other
    endpoint_scores = []
    for i in range(n_frames):
        # Sort similarities for this frame
        sims = sorted(similarity_matrix[i], reverse=True)
        # Endpoint should have sharp drop-off after nearby frames
        score = sims[10] - sims[50]  # Gap between close and far frames
        endpoint_scores.append(score)
    
    endpoint_candidates = np.argsort(endpoint_scores)[-5:]  # Top 5 candidates
    print(f"Heuristic 3 (endpoint candidates): {endpoint_candidates.tolist()}")
    
    # Use heuristic 2 as it's most reliable for sequential data
    optimal_start = best_start
    
    print(f"\n✅ Selected optimal start point: Frame {optimal_start}")
    return optimal_start

def greedy_order_frames(similarity_matrix, start_frame=None):
    """
    Order frames using greedy nearest neighbor approach.
    
    Args:
        similarity_matrix: NxN similarity matrix
        start_frame: Index of starting frame (None for auto-detection)
    
    Returns:
        List of frame indices in reconstructed order
    """
    n_frames = len(similarity_matrix)
    
    if start_frame is None:
        # Use smart start point detection
        start_frame = find_optimal_start_point(similarity_matrix)
    
    visited = set()
    frame_order = [start_frame]
    visited.add(start_frame)
    
    current = start_frame
    
    print(f"\nStarting from frame {start_frame}")
    print("Ordering frames using greedy algorithm...")
    
    for _ in tqdm(range(n_frames - 1), desc="Ordering frames"):
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
    similarity_file = "src/approach_resnet_tf_refinement/similarity_matrix.npy"
    output_file = "src/approach_resnet_tf_refinement/frame_order_initial.txt"
    
    print("Loading similarity matrix...")
    similarity_matrix = np.load(similarity_file)
    
    print("Computing optimal frame order with smart start detection...")
    frame_order = greedy_order_frames(similarity_matrix)
    
    save_frame_order(frame_order, output_file)
    print(f"Initial sequence length: {len(frame_order)}")
