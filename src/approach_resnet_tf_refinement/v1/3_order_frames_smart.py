import numpy as np
from tqdm import tqdm
import cv2
import os

def calculate_optical_flow_score(frame_paths, sequence):
    """
    Calculate optical flow consistency for a sequence.
    Returns flow consistency score (higher = better).
    """
    if len(sequence) < 3:
        return 0
    
    flows = []
    for i in range(min(10, len(sequence) - 1)):  # Test first 10 pairs
        idx1 = sequence[i]
        idx2 = sequence[i + 1]
        
        if idx1 >= len(frame_paths) or idx2 >= len(frame_paths):
            continue
        
        frame1 = cv2.imread(frame_paths[idx1])
        frame2 = cv2.imread(frame_paths[idx2])
        
        if frame1 is None or frame2 is None:
            continue
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        try:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            horizontal_flow = float(np.median(flow[..., 0]))
            flows.append(horizontal_flow)
        except:
            continue
    
    if not flows:
        return 0
    
    # Consistency = high mean + low variance
    consistency = abs(np.mean(flows)) - np.std(flows)
    return consistency


def find_optimal_start_point_v2(similarity_matrix, frame_paths=None, use_optical_flow=True):
    """
    Enhanced start point detection with optical flow validation.
    
    Args:
        similarity_matrix: NxN similarity matrix
        frame_paths: List of frame file paths (optional, for optical flow)
        use_optical_flow: Whether to use optical flow for validation
    
    Returns:
        Index of optimal starting frame
    """
    n_frames = len(similarity_matrix)
    
    print("\n=== Finding Optimal Start Point (Enhanced) ===")
    
    # Heuristic 1: Lowest average similarity (endpoints)
    avg_similarities = similarity_matrix.mean(axis=1)
    candidate1 = np.argmin(avg_similarities)
    print(f"Heuristic 1 (lowest avg similarity): Frame {candidate1} (avg sim: {avg_similarities[candidate1]:.4f})")
    
    # Heuristic 2: Asymmetric similarity distribution (edge detection)
    print("Heuristic 2: Edge detection...")
    edge_scores = []
    for i in range(n_frames):
        sims = similarity_matrix[i].copy()
        sims[i] = 0
        sorted_sims = np.sort(sims)[::-1]
        
        # Edge score: high similarity to nearby frames, low to distant
        top_30_avg = sorted_sims[:30].mean()
        bottom_30_avg = sorted_sims[-30:].mean()
        edge_score = top_30_avg - bottom_30_avg
        edge_scores.append(edge_score)
    
    edge_scores = np.array(edge_scores)
    top_edge_candidates = np.argsort(edge_scores)[-15:]  # Top 15 edge candidates
    print(f"Top edge candidates: {top_edge_candidates.tolist()}")
    
    # Heuristic 3: Best chain start (enhanced)
    print("Heuristic 3: Testing chain quality...")
    
    # Combine candidates from heuristic 1 and 2
    candidates_to_test = set(top_edge_candidates.tolist())
    candidates_to_test.add(candidate1)
    # Also add frames with very low average similarity
    low_sim_candidates = np.argsort(avg_similarities)[:20]
    candidates_to_test.update(low_sim_candidates.tolist())
    
    print(f"Testing {len(candidates_to_test)} candidate start points...")
    
    candidate_results = []
    
    for start_idx in tqdm(list(candidates_to_test), desc="Evaluating candidates"):
        visited = set([start_idx])
        current = start_idx
        sequence = [start_idx]
        chain_similarity = 0
        
        # Build longer chain with adaptive threshold
        similarity_threshold = 0.75  # Lower threshold (was 0.85)
        max_chain_length = min(80, n_frames - 1)  # Longer test (was 50)
        
        for step in range(max_chain_length):
            similarities = similarity_matrix[current].copy()
            similarities[list(visited)] = -1
            
            next_frame = np.argmax(similarities)
            max_sim = similarities[next_frame]
            
            # Adaptive threshold: allow slightly lower similarity as chain grows
            adjusted_threshold = similarity_threshold - (step * 0.001)
            
            if max_sim < adjusted_threshold:
                break
            
            chain_similarity += max_sim
            visited.add(next_frame)
            sequence.append(next_frame)
            current = next_frame
        
        # Calculate multiple quality metrics
        chain_length = len(sequence)
        avg_chain_similarity = chain_similarity / max(1, chain_length - 1)
        
        # Optical flow consistency (if available)
        flow_score = 0
        if use_optical_flow and frame_paths is not None:
            flow_score = calculate_optical_flow_score(frame_paths, sequence)
        
        # Edge score for this candidate
        candidate_edge_score = edge_scores[start_idx]
        
        # Combined score
        # - Longer chains are better
        # - Higher average similarity is better
        # - Good optical flow consistency is better
        # - Being an edge candidate is better
        combined_score = (
            chain_length * 2.0 +                    # Chain length (most important)
            avg_chain_similarity * 50.0 +           # Average similarity
            flow_score * 5.0 +                       # Flow consistency
            candidate_edge_score * 10.0              # Edge score
        )
        
        candidate_results.append({
            'frame': start_idx,
            'score': combined_score,
            'chain_length': chain_length,
            'avg_similarity': avg_chain_similarity,
            'flow_score': flow_score,
            'edge_score': candidate_edge_score
        })
    
    # Sort by combined score
    candidate_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Show top 5 candidates
    print("\n📊 Top 5 Candidates:")
    for i, result in enumerate(candidate_results[:5], 1):
        print(f"  {i}. Frame {result['frame']:3d}: "
              f"score={result['score']:7.2f}, "
              f"chain={result['chain_length']:3d}, "
              f"avg_sim={result['avg_similarity']:.4f}, "
              f"flow={result['flow_score']:6.2f}, "
              f"edge={result['edge_score']:.4f}")
    
    # Select best candidate
    optimal_start = candidate_results[0]['frame']
    
    # Additional validation: Check if reversing would be better
    if use_optical_flow and frame_paths is not None:
        print("\n🔄 Validating direction...")
        best_candidate = candidate_results[0]
        
        # Build sequence from best candidate
        visited = set([optimal_start])
        current = optimal_start
        test_sequence = [optimal_start]
        
        for _ in range(min(30, n_frames - 1)):
            similarities = similarity_matrix[current].copy()
            similarities[list(visited)] = -1
            next_frame = np.argmax(similarities)
            if similarities[next_frame] < 0.7:
                break
            test_sequence.append(next_frame)
            visited.add(next_frame)
            current = next_frame
        
        # Test flow in both directions
        forward_flow = calculate_optical_flow_score(frame_paths, test_sequence)
        backward_flow = calculate_optical_flow_score(frame_paths, test_sequence[::-1])
        
        print(f"  Forward flow score: {forward_flow:.3f}")
        print(f"  Backward flow score: {backward_flow:.3f}")
        
        if backward_flow > forward_flow * 1.2:  # Significantly better backward
            optimal_start = test_sequence[-1]
            print(f"  ⚠️  Backward is better! Using frame {optimal_start} instead")
    
    print(f"\n✅ Selected optimal start point: Frame {optimal_start}")
    print(f"   (Chain length: {candidate_results[0]['chain_length']}, Score: {candidate_results[0]['score']:.2f})")
    
    return optimal_start


def greedy_order_frames(similarity_matrix, start_frame=None, frame_paths=None):
    """
    Order frames using greedy nearest neighbor approach with enhanced start detection.
    
    Args:
        similarity_matrix: NxN similarity matrix
        start_frame: Index of starting frame (None for auto-detection)
        frame_paths: List of frame file paths (for optical flow validation)
    
    Returns:
        List of frame indices in reconstructed order
    """
    n_frames = len(similarity_matrix)
    
    if start_frame is None:
        # Use enhanced start point detection
        use_flow = frame_paths is not None
        start_frame = find_optimal_start_point_v2(similarity_matrix, frame_paths, use_flow)
    
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
    similarity_file = "similarity_matrix.npy"
    output_file = "frame_order_initial.txt"
    frames_dir = "../../../frames"  # Directory containing frame images
    
    print("Loading similarity matrix...")
    similarity_matrix = np.load(similarity_file)
    
    # Get frame paths for optical flow
    print("Loading frame paths...")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))])
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    print(f"Found {len(frame_paths)} frames")
    
    print("\nComputing optimal frame order with enhanced start detection...")
    frame_order = greedy_order_frames(similarity_matrix, frame_paths=frame_paths)
    
    save_frame_order(frame_order, output_file)
    print(f"\n✅ Initial sequence length: {len(frame_order)}")