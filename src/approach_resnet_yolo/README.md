# ResNet50 + YOLOv11 Hybrid Approach

## Overview

This approach combines deep learning features from two complementary models:
- **ResNet50**: Extracts high-level semantic features (2048-dimensional embeddings)
- **YOLOv11**: Detects objects and extracts object-based features using the latest YOLO model

By combining both semantic understanding and object-level information, this hybrid approach aims to achieve better frame ordering accuracy.

## Approach Pipeline

### 1. Extract ResNet50 Features (`1_extract_resnet_features.py`)
- Loads pre-trained ResNet50 model
- Removes the final classification layer to get feature embeddings
- Extracts 2048-dimensional feature vectors for each frame
- Saves features to `resnet_features.npy`

### 2. Extract YOLOv11 Features (`2_extract_yolo_features.py`)
- Loads pre-trained YOLOv11x (extra-large) model for maximum accuracy
- Detects objects in each frame using state-of-the-art detection
- Extracts enhanced features based on:
  - Object class distribution (80 COCO classes)
  - Extended spatial information (bbox positions, sizes, dispersion)
  - Detection confidence scores
  - Object count and area coverage
- Feature dimension: 88 (80 classes + 8 spatial features)
- Saves features to `yolo_features.npy`

### 3. Compute Combined Similarity (`3_compute_similarity.py`)
- Computes cosine similarity for ResNet features
- Computes cosine similarity for YOLO features
- Combines similarities with weighted sum:
  ```
  S_ij = α × ResNet_similarity + β × YOLO_similarity + γ × motion_consistency
  ```
- Default weights: α=0.6, β=0.3, γ=0.1
- Saves combined similarity matrix to `combined_similarity.npy`

### 4. Order Frames (`4_order_frames.py`)
- Uses greedy nearest neighbor algorithm
- Starts from frame with highest total similarity
- At each step, selects unvisited frame with highest similarity
- Saves frame order to `frame_order.txt`

### 5. Reconstruct Video (`5_reconstruct_video.py`)
- Reads frames in computed order
- Writes reconstructed video at 30 FPS
- Saves output to `output/reconstructed_resnet_yolo.mp4`

## Usage

```bash
# Step 1: Extract ResNet50 features
python src/approach_resnet_yolo/1_extract_resnet_features.py

# Step 2: Extract YOLO features
python src/approach_resnet_yolo/2_extract_yolo_features.py

# Step 3: Compute combined similarity
python src/approach_resnet_yolo/3_compute_similarity.py

# Step 4: Order frames
python src/approach_resnet_yolo/4_order_frames.py

# Step 5: Reconstruct video
python src/approach_resnet_yolo/5_reconstruct_video.py
```

## Dependencies

```bash
pip install torch torchvision opencv-python numpy scikit-learn tqdm pillow ultralytics>=8.3.0
```

Note: YOLOv11 requires ultralytics>=8.3.0

## Advantages

1. **Semantic Understanding**: ResNet50 captures high-level visual semantics
2. **State-of-the-art Object Detection**: YOLOv11x provides the most accurate object detection available
3. **Enhanced Spatial Features**: 8-dimensional spatial representation captures object distribution and movement
4. **Complementary Features**: Combines global scene understanding with local object details
5. **Robustness**: Less sensitive to lighting changes compared to pixel-based methods
6. **Latest Technology**: Uses cutting-edge YOLOv11 architecture with improved accuracy

## Design Considerations

- **α (ResNet weight)**: Higher value prioritizes semantic similarity
- **β (YOLO weight)**: Higher value prioritizes object consistency
- **γ (Motion weight)**: Reserved for future optical flow integration
- **Trade-off**: Balance between accuracy and computational cost

## Future Improvements

1. Add optical flow for motion consistency (γ term)
2. Implement A* or beam search for better ordering
3. Add temporal smoothing for local refinement
4. Experiment with different similarity metrics (Euclidean, Manhattan)
5. Fine-tune YOLOv11 on specific domain if needed
6. Explore YOLOv11 segmentation features for even richer representation
