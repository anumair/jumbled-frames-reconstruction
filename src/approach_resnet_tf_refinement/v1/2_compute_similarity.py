import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def compute_similarity_matrix(features_file, output_file):
    """
    Compute cosine similarity matrix from ResNet features.
    
    Args:
        features_file: Path to features (.npy)
        output_file: Path to save similarity matrix (.npy)
    """
    print("Loading ResNet features...")
    features = np.load(features_file)
    n_frames = len(features)
    print(f"Computing similarity for {n_frames} frames...")
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(features)
    
    # Save similarity matrix
    np.save(output_file, similarity_matrix)
    print(f"✅ Similarity matrix saved to '{output_file}'")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    
    return similarity_matrix

if __name__ == "__main__":
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    features_file = os.path.join(project_root, "src", "approach_resnet_tf_refinement", "resnet_features.npy")
    output_file = os.path.join(project_root, "src", "approach_resnet_tf_refinement", "similarity_matrix.npy")
    
    compute_similarity_matrix(features_file, output_file)
