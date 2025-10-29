import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def extract_yolo_features(frames_dir, output_file):
    """
    Extract YOLO object detection features from all frames.
    Uses YOLOv11 (latest) to detect objects and extract embeddings.
    
    Args:
        frames_dir: Directory containing extracted frames
        output_file: Path to save extracted features (.npy)
    """
    # Load YOLOv11 model (most advanced version)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Using YOLOv11x (extra-large) for best accuracy
    # You can also use: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    model = YOLO('yolo11x.pt')  # Will auto-download if not present
    model.to(device)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    print(f"Found {len(frame_files)} frames")
    
    features = []
    
    with torch.no_grad():
        for frame_file in tqdm(frame_files, desc="Extracting YOLOv11 features"):
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Run YOLOv11 inference
            results = model(frame_path, verbose=False)
            
            # Extract features from detections
            # We'll create a feature vector based on:
            # 1. Number of detections per class
            # 2. Bounding box positions and sizes
            # 3. Confidence scores
            # 4. Object tracking features
            
            # Get detection results
            boxes = results[0].boxes
            
            # Create feature vector (80 classes for COCO dataset + spatial features)
            feature_vector = np.zeros(80 + 8)  # 80 classes + 8 enhanced spatial features
            
            if boxes is not None and len(boxes) > 0:
                # Extract data from boxes
                xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                conf = boxes.conf.cpu().numpy()  # Confidence scores
                cls = boxes.cls.cpu().numpy()    # Class IDs
                
                # Count detections per class (weighted by confidence)
                for i in range(len(boxes)):
                    class_id = int(cls[i])
                    confidence = conf[i]
                    feature_vector[class_id] += confidence
                
                # Add enhanced spatial information
                img = cv2.imread(frame_path)
                img_h, img_w = img.shape[:2]
                
                centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
                centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
                widths = xyxy[:, 2] - xyxy[:, 0]
                heights = xyxy[:, 3] - xyxy[:, 1]
                areas = widths * heights
                
                # Normalized spatial features
                feature_vector[80] = np.mean(centers_x) / img_w  # Avg center X
                feature_vector[81] = np.mean(centers_y) / img_h  # Avg center Y
                feature_vector[82] = np.mean(widths) / img_w     # Avg width
                feature_vector[83] = np.mean(heights) / img_h    # Avg height
                feature_vector[84] = np.std(centers_x) / img_w   # Std center X (dispersion)
                feature_vector[85] = np.std(centers_y) / img_h   # Std center Y
                feature_vector[86] = np.mean(areas) / (img_w * img_h)  # Avg area
                feature_vector[87] = len(boxes) / 10.0           # Number of objects (normalized)
            
            features.append(feature_vector)
    
    features = np.array(features)
    np.save(output_file, features)
    print(f"✅ YOLOv11 features saved to '{output_file}'")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dimension: {features.shape[1]} (80 classes + 8 spatial features)")

if __name__ == "__main__":
    frames_dir = "frames"
    output_file = "src/approach_resnet_yolo/yolo_features.npy"
    extract_yolo_features(frames_dir, output_file)
