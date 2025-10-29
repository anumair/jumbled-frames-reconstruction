import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def compute_combined_similarity(resnet_features_file, yolo_features_file, output_file, 
                                alpha=0.6, beta=0.3, gamma=0.1):
    """
    Compute combined similarity matrix using ResNet and YOLO features.
    
    Args:
        resnet_features_file: Path to ResNet features (.npy)
        yolo_features_file: Path to YOLO features (.npy)
        output_file: Path to save similarity matrix (.npy)
        alpha: Weight for ResNet semantic similarity
        beta: Weight for YOLO object similarity
        gamma: Weight for motion consistency (placeholder for now)
    """
    print("Loading features...")
    resnet_features = np.load(resnet_features_file)
    yolo_features = np.load(yolo_features_file)
    
    n_frames = len(resnet_features)
    print(f"Computing similarity for {n_frames} frames...")
    
    # Compute ResNet similarity (semantic)
    print("Computing ResNet semantic similarity...")
    resnet_similarity = cosine_similarity(resnet_features)
    
    # Compute YOLO similarity (object-based)
    print("Computing YOLO object similarity...")
    yolo_similarity = cosine_similarity(yolo_features)
    
    # Combine similarities
    print(f"Combining similarities with weights: alpha={alpha}, beta={beta}, gamma={gamma}")
    combined_similarity = (alpha * resnet_similarity + 
                          beta * yolo_similarity)
    
    # TODO: Add motion consistency term (gamma)
    # This could be based on optical flow or temporal coherence
    
    # Normalize to [0, 1]
    combined_similarity = (combined_similarity - combined_similarity.min()) / (combined_similarity.max() - combined_similarity.min())
    
    # Save similarity matrix
    np.save(output_file, combined_similarity)
    print(f"✅ Combined similarity matrix saved to '{output_file}'")
    print(f"Similarity matrix shape: {combined_similarity.shape}")
    print(f"Similarity range: [{combined_similarity.min():.4f}, {combined_similarity.max():.4f}]")

if __name__ == "__main__":
    resnet_features_file = "src/approach_resnet_yolo/resnet_features.npy"
    yolo_features_file = "src/approach_resnet_yolo/yolo_features.npy"
    output_file = "src/approach_resnet_yolo/combined_similarity.npy"
    
    compute_combined_similarity(resnet_features_file, yolo_features_file, output_file)
