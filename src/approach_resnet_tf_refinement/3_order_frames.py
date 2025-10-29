import numpy as np
from tqdm import tqdm

def find_optimal_start_point(similarity_matrix):
    """
    Find the optimal starting point using multiple advanced heuristics.
    
    Args:
        similarity_matrix: NxN similarity matrix
    
    Returns:
        Index of optimal starting frame
    """
    n_frames = len(similarity_matrix)
    
    print("\n=== Finding Optimal Start Point (Advanced) ===")
    
    # Heuristic 1: Find endpoints using asymmetric similarity
    # Start/end frames have high similarity in one direction, low in the other
    print("\nHeuristic 1: Asymmetric similarity analysis...")
    endpoint_scores = []
    for i in range(n_frames):
        # Get top similar frames
        sorted_sims = np.sort(similarity_matrix[i])[::-1]
        # Calculate asymmetry: high local similarity but low global
        local_sim = np.mean(sorted_sims[1:6])  # Top 5 neighbors
        global_sim = np.mean(sorted_sims[20:])  # Distant frames
        asymmetry = local_sim - global_sim
        endpoint_scores.append(asymmetry)
    
    # Get top endpoint candidates
    endpoint_candidates = np.argsort(endpoint_scores)[-10:]
    print(f"Top 10 endpoint candidates: {endpoint_candidates.tolist()}")
    print(f"Their scores: {[f'{endpoint_scores[i]:.4f}' for i in endpoint_candidates]}")
    
    # Heuristic 2: Build forward chains from each endpoint candidate
    print("\nHeuristic 2: Forward chain quality from endpoints...")
    best_start = endpoint_candidates[0]
    best_chain_quality = 0
    
    for candidate in tqdm(endpoint_candidates, desc="Testing candidates"):
        visited = set([candidate])
        current = candidate
        chain_quality = 0
        consistency_score = 0
        
        # Build longer chain (first 100 frames)
        prev_similarity = 1.0
        for step in range(min(100, n_frames-1)):
            similarities = similarity_matrix[current].copy()
            similarities[list(visited)] = -1
            
            next_frame = np.argmax(similarities)
            next_sim = similarities[next_frame]
            
            if next_sim < 0.8:  # Lower threshold
                break
            
            # Reward smooth transitions
            consistency = 1.0 - abs(next_sim - prev_similarity)
            chain_quality += next_sim * (1 + consistency)
            consistency_score += consistency
            
            visited.add(next_frame)
            current = next_frame
            prev_similarity = next_sim
        
        # Penalize short chains
        chain_length = len(visited)
        if chain_length > best_chain_quality / 2:  # Prefer longer chains
            quality = chain_quality * (chain_length / 100.0) * (1 + consistency_score / chain_length)
            if quality > best_chain_quality:
                best_chain_quality = quality
                best_start = candidate
                print(f"  Frame {candidate}: quality={quality:.2f}, length={chain_length}, avg_consistency={consistency_score/chain_length:.4f}")
    
    # Heuristic 3: Directional flow analysis
    print("\nHeuristic 3: Directional flow analysis...")
    flow_scores = []
    for i in range(n_frames):
        # Check if similarities decrease as we move "forward"
        sims = similarity_matrix[i]
        forward_decrease = 0
        for j in range(1, min(50, n_frames)):
            if i + j < n_frames:
                forward_decrease += max(0, sims[i+j-1] - sims[i+j])
        flow_scores.append(forward_decrease)
    
    flow_candidate = np.argmax(flow_scores)
    print(f"Flow analysis suggests: Frame {flow_candidate} (score: {flow_scores[flow_candidate]:.4f})")
    
    # Heuristic 4: Clustering-based approach
    print("\nHeuristic 4: Temporal cluster boundary...")
    # Find frames with low similarity to previous cluster
    cluster_boundary_scores = []
    window = 20
    for i in range(n_frames):
        # Compare similarity to frames before vs after
        before_indices = list(range(max(0, i-window), i))
        after_indices = list(range(i+1, min(n_frames, i+window+1)))
        
        if before_indices and after_indices:
            sim_before = np.mean([similarity_matrix[i, j] for j in before_indices])
            sim_after = np.mean([similarity_matrix[i, j] for j in after_indices])
            boundary_score = sim_after - sim_before  # Start has higher similarity to future
        else:
            boundary_score = 0
        cluster_boundary_scores.append(boundary_score)
    
    cluster_candidate = np.argmax(cluster_boundary_scores)
    print(f"Cluster boundary suggests: Frame {cluster_candidate} (score: {cluster_boundary_scores[cluster_candidate]:.4f})")
    
    # Final decision: Weight the heuristics
    print("\n=== Final Decision ===")
    candidates = {
        best_start: best_chain_quality,
        flow_candidate: flow_scores[flow_candidate] * 10,
        cluster_candidate: cluster_boundary_scores[cluster_candidate] * 100
    }
    
    print("Weighted candidates:")
    for cand, score in candidates.items():
        print(f"  Frame {cand}: weighted_score={score:.2f}")
    
    # Choose the one with best chain quality (most reliable)
    optimal_start = best_start
    
    print(f"\n✅ Selected optimal start: Frame {optimal_start}")
    print(f"   (Reason: Best forward chain quality)")
    
    return optimal_start

def greedy_order_frames(similarity_matrix, start_frame=None, smart_start=True):
    """
    Order frames using greedy nearest neighbor approach.
    
    Args:
        similarity_matrix: NxN similarity matrix
        start_frame: Index of starting frame (None for auto-selection)
        smart_start: Use smart start detection (default: True)
    
    Returns:
        List of frame indices in reconstructed order
    """
    n_frames = len(similarity_matrix)
    
    if start_frame is None:
        if smart_start:
            # Use smart start point detection
            start_frame = find_optimal_start_point(similarity_matrix)
        else:
            # Start from the frame with highest total similarity
            start_frame = np.argmax(similarity_matrix.sum(axis=1))
    
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
    
    print("Computing optimal frame order...")
    frame_order = greedy_order_frames(similarity_matrix)
    
    save_frame_order(frame_order, output_file)
    print(f"Initial sequence length: {len(frame_order)}")
