import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def extract_resnet_features_pytorch(frames_dir, output_file):
    """
    Extract ResNet50 features using PyTorch.
    
    Args:
        frames_dir: Directory containing frame images
        output_file: Path to save features (.npy)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained ResNet50 model
    print("Loading ResNet50 model from PyTorch...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Remove the final classification layer to get feature embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Output feature dimension: 2048")
    
    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    n_frames = len(frame_files)
    print(f"Found {n_frames} frames")
    
    # Extract features
    features = []
    print("Extracting ResNet50 features...")
    
    with torch.no_grad():
        for frame_file in tqdm(frame_files, desc="Processing frames"):
            # Load and preprocess image
            img_path = os.path.join(frames_dir, frame_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Extract features
            feature = model(img_tensor)
            feature = feature.squeeze().cpu().numpy()
            features.append(feature)
    
    # Convert to numpy array
    features = np.array(features)
    
    # Save features
    np.save(output_file, features)
    print(f"✅ ResNet50 features saved to '{output_file}'")
    print(f"Feature shape: {features.shape}")

if __name__ == "__main__":
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "frames")
    output_file = os.path.join(project_root, "src", "approach_resnet_tf_refinement", "resnet_features.npy")
    
    extract_resnet_features_pytorch(frames_dir, output_file)
