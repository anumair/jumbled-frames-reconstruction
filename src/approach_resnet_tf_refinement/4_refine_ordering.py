import numpy as np
from tqdm import tqdm

def calculate_path_cost(frame_order, similarity_matrix):
    """Calculate total cost (negative similarity) of a path."""
    cost = 0
    for i in range(len(frame_order) - 1):
        cost += similarity_matrix[frame_order[i], frame_order[i+1]]
    return cost

def two_opt_swap(frame_order, i, k):
    """Perform 2-opt swap: reverse the segment between i and k."""
    new_order = frame_order[:i] + frame_order[i:k+1][::-1] + frame_order[k+1:]
    return new_order

def refine_with_2opt(frame_order, similarity_matrix, max_iterations=100):
    """
    Refine frame ordering using 2-opt local search.
    This swaps segments to find local improvements.
    
    Args:
        frame_order: Initial frame order
        similarity_matrix: Similarity matrix
        max_iterations: Maximum number of iterations
    
    Returns:
        Refined frame order
    """
    n_frames = len(frame_order)
    current_order = frame_order.copy()
    current_cost = calculate_path_cost(current_order, similarity_matrix)
    
    print(f"Initial path cost (total similarity): {current_cost:.4f}")
    print("Refining with 2-opt algorithm...")
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(1, n_frames - 2):
            for k in range(i + 1, n_frames - 1):
                # Try 2-opt swap
                new_order = two_opt_swap(current_order, i, k)
                new_cost = calculate_path_cost(new_order, similarity_matrix)
                
                # If improvement found, accept it
                if new_cost > current_cost:
                    current_order = new_order
                    current_cost = new_cost
                    improved = True
                    print(f"Iteration {iteration}: Improved cost to {current_cost:.4f}")
                    break
            
            if improved:
                break
    
    print(f"Final path cost: {current_cost:.4f}")
    print(f"Improvement: {current_cost - calculate_path_cost(frame_order, similarity_matrix):.4f}")
    
    return current_order

def refine_with_sliding_window(frame_order, similarity_matrix, window_size=5):
    """
    Refine ordering using sliding window local optimization.
    For each window, try all permutations to find the best local order.
    
    Args:
        frame_order: Initial frame order
        similarity_matrix: Similarity matrix
        window_size: Size of sliding window
    
    Returns:
        Refined frame order
    """
    from itertools import permutations
    
    n_frames = len(frame_order)
    current_order = frame_order.copy()
    
    print(f"Refining with sliding window (window_size={window_size})...")
    
    for start_idx in tqdm(range(n_frames - window_size + 1), desc="Sliding window"):
        window = current_order[start_idx:start_idx + window_size]
        
        # Try all permutations of the window
        best_window = window
        best_cost = -float('inf')
        
        for perm in permutations(window):
            # Calculate cost of this permutation
            cost = 0
            for i in range(len(perm) - 1):
                cost += similarity_matrix[perm[i], perm[i+1]]
            
            if cost > best_cost:
                best_cost = cost
                best_window = list(perm)
        
        # Update the order with best window
        current_order[start_idx:start_idx + window_size] = best_window
    
    return current_order

def save_frame_order(frame_order, output_file):
    """Save frame order to text file."""
    with open(output_file, 'w') as f:
        for idx in frame_order:
            f.write(f"{idx}\n")
    print(f"✅ Refined frame order saved to '{output_file}'")

if __name__ == "__main__":
    # Load initial frame order
    initial_order_file = "src/approach_resnet_tf_refinement/frame_order_initial.txt"
    similarity_file = "src/approach_resnet_tf_refinement/similarity_matrix.npy"
    output_file = "src/approach_resnet_tf_refinement/frame_order_refined.txt"
    
    print("Loading initial frame order...")
    with open(initial_order_file, 'r') as f:
        frame_order = [int(line.strip()) for line in f.readlines()]
    
    print("Loading similarity matrix...")
    similarity_matrix = np.load(similarity_file)
    
    # Method 1: 2-opt refinement
    print("\n=== Method 1: 2-opt Refinement ===")
    refined_order_2opt = refine_with_2opt(frame_order, similarity_matrix, max_iterations=50)
    
    # Method 2: Sliding window (use smaller window for speed)
    print("\n=== Method 2: Sliding Window Refinement ===")
    refined_order_window = refine_with_sliding_window(refined_order_2opt, similarity_matrix, window_size=4)
    
    # Save the final refined order
    save_frame_order(refined_order_window, output_file)
    print(f"\nRefined sequence length: {len(refined_order_window)}")
    
    # Calculate improvements
    initial_cost = calculate_path_cost(frame_order, similarity_matrix)
    final_cost = calculate_path_cost(refined_order_window, similarity_matrix)
    print(f"\nInitial total similarity: {initial_cost:.4f}")
    print(f"Final total similarity: {final_cost:.4f}")
    print(f"Total improvement: {final_cost - initial_cost:.4f} ({((final_cost - initial_cost) / initial_cost * 100):.2f}%)")
