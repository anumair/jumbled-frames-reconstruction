import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def extract_resnet_features(frames_dir, output_file):
    """
    Extract ResNet50 features from all frames.
    
    Args:
        frames_dir: Directory containing extracted frames
        output_file: Path to save extracted features (.npy)
    """
    # Load pre-trained ResNet50 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer
    resnet = resnet.to(device)
    resnet.eval()
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    print(f"Found {len(frame_files)} frames")
    
    features = []
    
    with torch.no_grad():
        for frame_file in tqdm(frame_files, desc="Extracting ResNet50 features"):
            frame_path = os.path.join(frames_dir, frame_file)
            image = Image.open(frame_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # Extract features
            feature = resnet(image_tensor)
            feature = feature.cpu().numpy().flatten()
            features.append(feature)
    
    features = np.array(features)
    np.save(output_file, features)
    print(f"✅ ResNet50 features saved to '{output_file}'")
    print(f"Feature shape: {features.shape}")

if __name__ == "__main__":
    frames_dir = "frames"
    output_file = "src/approach_resnet_yolo/resnet_features.npy"
    extract_resnet_features(frames_dir, output_file)
