"""
ResNet50 Feature Extraction for Frame Reconstruction

This script extracts deep learning features from video frames using a pre-trained ResNet50 model.
Unlike ORB features, ResNet50 captures high-level semantic information that's more robust to
lighting changes and provides better similarity measurements.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_resnet50_model():
    """
    Load pre-trained ResNet50 model and remove the final classification layer.
    We'll use the penultimate layer (2048-dim features) for similarity comparison.
    """
    print("Loading pre-trained ResNet50 model...")
    model = models.resnet50(pretrained=True)
    
    # Remove the final fully connected layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"✅ Model loaded on device: {device}")
    return model, device


def get_image_transform():
    """
    Define image preprocessing pipeline for ResNet50.
    ResNet50 expects 224x224 RGB images with specific normalization.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])
    return transform


def extract_frame_features(frames_dir, output_path="resnet50_features.npy"):
    """
    Extract ResNet50 features for all frames in the directory.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Path to save the feature matrix
        
    Returns:
        features: numpy array of shape (N, 2048) where N is number of frames
    """
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    n_frames = len(frame_files)
    
    print(f"Found {n_frames} frames to process")
    
    # Load model and preprocessing
    model, device = load_resnet50_model()
    transform = get_image_transform()
    
    # Initialize feature matrix (2048-dimensional features from ResNet50)
    features = np.zeros((n_frames, 2048), dtype=np.float32)
    
    print("\nExtracting ResNet50 features from frames...")
    
    with torch.no_grad():  # Disable gradient computation for inference
        for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
            # Load and preprocess image
            img_path = os.path.join(frames_dir, frame_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract features
            feature = model(img_tensor)
            
            # Flatten and convert to numpy
            feature = feature.squeeze().cpu().numpy()
            features[idx] = feature
    
    # Save features
    np.save(output_path, features)
    print(f"\n✅ Features saved to: {output_path}")
    print(f"Feature matrix shape: {features.shape}")
    
    return features


def compute_cosine_similarity_matrix(features, output_path="similarity_matrix_resnet50.npy"):
    """
    Compute cosine similarity matrix from ResNet50 features.
    Cosine similarity is better for high-dimensional embeddings than Euclidean distance.
    
    Args:
        features: numpy array of shape (N, 2048)
        output_path: Path to save similarity matrix
        
    Returns:
        similarity_matrix: numpy array of shape (N, N)
    """
    print("\nComputing cosine similarity matrix...")
    
    n_frames = features.shape[0]
    
    # Normalize features (for cosine similarity)
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
    
    # Compute similarity matrix (dot product of normalized features)
    similarity_matrix = np.dot(features_norm, features_norm.T)
    
    # Ensure values are in [0, 1] range (handle numerical errors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)
    
    # Save similarity matrix
    np.save(output_path, similarity_matrix)
    print(f"✅ Similarity matrix saved to: {output_path}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return similarity_matrix


if __name__ == "__main__":
    # Configuration
    frames_dir = "frames"
    features_output = "src/approach_resnet50/resnet50_features.npy"
    similarity_output = "src/approach_resnet50/similarity_matrix_resnet50.npy"
    
    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        print(f"❌ Error: Frames directory not found at {frames_dir}")
        print("Please run extract_frames.py first!")
        exit(1)
    
    # Extract features
    features = extract_frame_features(frames_dir, features_output)
    
    # Compute similarity matrix
    similarity_matrix = compute_cosine_similarity_matrix(features, similarity_output)
    
    print("\n" + "="*60)
    print("ResNet50 Feature Extraction Complete!")
    print("="*60)
