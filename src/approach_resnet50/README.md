# ResNet50-Based Frame Reconstruction Approach

## Overview

This approach uses **deep learning features** extracted from a pre-trained **ResNet50** model to reconstruct the jumbled video. Unlike traditional feature matching (like ORB), ResNet50 captures high-level semantic information that is more robust to lighting changes, motion blur, and provides better similarity measurements for complex scenes.

---

## Why ResNet50?

### Advantages over ORB:
✅ **Semantic Understanding:** Captures high-level features (objects, textures, patterns) rather than just keypoints  
✅ **Robustness:** More resistant to lighting changes, rotation, and scale variations  
✅ **Consistent Features:** Produces 2048-dimensional feature vectors for every image  
✅ **Pre-trained on ImageNet:** Leverages knowledge from millions of images  
✅ **Better Generalization:** Works well even on scenes without distinct keypoints  

---

## Methodology

### Stage 1: Feature Extraction

**File:** `extract_features_resnet50.py`

Extract deep learning features from each frame using pre-trained ResNet50.

**Process:**
1. **Load Pre-trained Model:**
   - ResNet50 trained on ImageNet dataset
   - Remove final classification layer
   - Use penultimate layer for 2048-dim feature vectors

2. **Image Preprocessing:**
   - Resize images to 256×256
   - Center crop to 224×224 (ResNet50 input size)
   - Normalize using ImageNet mean and std

3. **Feature Extraction:**
   - Pass each frame through the network
   - Extract 2048-dimensional feature vector
   - Store features in numpy array (300 × 2048)

4. **Similarity Computation:**
   - Normalize feature vectors
   - Compute **cosine similarity** between all pairs
   - Cosine similarity is ideal for high-dimensional embeddings
   - Create 300×300 similarity matrix

**Key Insight:**
ResNet50 features capture semantic similarity. Consecutive frames will have high cosine similarity due to similar scene content.

**Usage:**
```bash
python src/approach_resnet50/extract_features_resnet50.py
```

**Output:**
- `resnet50_features.npy` (~2.4 MB, 300×2048 features)
- `similarity_matrix_resnet50.npy` (~352 KB, 300×300 similarities)

---

### Stage 2: Frame Ordering

**File:** `order_frames_resnet50.py`

Order frames using the ResNet50 similarity matrix.

**Algorithm:**

#### 2.1 Greedy Ordering
```
1. Start from frame with lowest average similarity (likely first frame)
2. At each step:
   - Look at similarity scores from current frame to all unvisited frames
   - Pick the frame with highest similarity
   - Mark it as visited and move to it
3. Continue until all frames are ordered
```

#### 2.2 2-Opt Local Optimization
Apply classic 2-opt algorithm to improve ordering:
- Try reversing segments of the path
- Keep reversals that increase total similarity
- Iterate until no more improvements found
- Provides significant quality boost

**Usage:**
```bash
python src/approach_resnet50/order_frames_resnet50.py
```

**Output:**
- `frame_order_resnet50.txt` (300 frame indices)
- `output/reconstructed_resnet50.mp4` (~62 MB)
- `execution_log_resnet50.txt` (performance metrics)

---

### Stage 3: Video Reconstruction

**Integrated in:** `order_frames_resnet50.py`

Reconstruct video from ordered frames:
1. Read frames in computed order
2. Write to video using OpenCV VideoWriter
3. Output at 30 FPS, 1080p resolution

---

## Design Considerations

| Aspect | Implementation Detail |
|--------|----------------------|
| **Model** | ResNet50 pre-trained on ImageNet |
| **Feature Dimension** | 2048-dimensional feature vectors |
| **Similarity Metric** | Cosine similarity (better for high-dim embeddings) |
| **Ordering Algorithm** | Greedy + 2-opt local search |
| **Time Complexity** | O(N) for feature extraction, O(N²) for ordering |
| **GPU Support** | Automatic GPU usage if CUDA available |
| **Memory Efficiency** | Features stored in float32 format |

---

## Comparison with ORB Approach

| Feature | ORB Approach | ResNet50 Approach |
|---------|--------------|-------------------|
| **Feature Type** | Local keypoints (ORB) | Global semantic features (CNN) |
| **Dimension** | Variable (depends on keypoints) | Fixed 2048-dim vectors |
| **Robustness** | Sensitive to lighting/blur | More robust to variations |
| **Computation** | Faster (~5-10 min) | Slower (~15-30 min with GPU) |
| **Accuracy** | Good for textured scenes | Better for complex scenes |
| **Dependencies** | OpenCV only | Requires PyTorch |
| **Model Size** | N/A | ~100 MB (ResNet50) |

---

## Dependencies

```
# Core
opencv-python>=4.8.0
numpy>=1.24.0
tqdm>=4.65.0
pillow>=10.0.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
```

**Install with:**
```bash
pip install torch torchvision opencv-python numpy tqdm pillow
```

**Note:** For GPU acceleration, install CUDA-compatible PyTorch from [pytorch.org](https://pytorch.org)

---

## File Structure

```
src/approach_resnet50/
├── extract_features_resnet50.py       # Feature extraction & similarity
├── order_frames_resnet50.py           # Frame ordering & reconstruction
├── resnet50_features.npy              # Extracted features (generated)
├── similarity_matrix_resnet50.npy     # Similarity matrix (generated)
└── README.md                          # This file
```

---

## How to Run

### Complete Pipeline:

```bash
# Step 1: Extract ResNet50 features and compute similarity
python src/approach_resnet50/extract_features_resnet50.py

# Step 2: Order frames and reconstruct video
python src/approach_resnet50/order_frames_resnet50.py
```

### Quick Run (if features exist):

```bash
# Just run ordering and reconstruction
python src/approach_resnet50/order_frames_resnet50.py
```

---

## Output Files

- **Features:** `src/approach_resnet50/resnet50_features.npy` (~2.4 MB)
- **Similarity Matrix:** `src/approach_resnet50/similarity_matrix_resnet50.npy` (~352 KB)
- **Frame Order:** `frame_order_resnet50.txt` (300 lines)
- **Reconstructed Video:** `output/reconstructed_resnet50.mp4` (~62 MB, 10 sec @ 30 FPS)
- **Execution Log:** `execution_log_resnet50.txt`

---

## Performance Metrics

### With GPU (CUDA):
- **Feature Extraction:** ~2-5 minutes
- **Similarity Computation:** ~1-2 seconds
- **Frame Ordering:** ~10-20 seconds
- **Video Reconstruction:** ~5-10 seconds
- **Total Time:** ~3-7 minutes

### With CPU:
- **Feature Extraction:** ~10-20 minutes
- **Similarity Computation:** ~5-10 seconds
- **Frame Ordering:** ~10-20 seconds
- **Video Reconstruction:** ~5-10 seconds
- **Total Time:** ~11-21 minutes

---

## Advantages

✅ **High Accuracy:** Deep features capture semantic similarity better than local features  
✅ **Robustness:** Handles lighting changes, motion blur, and scale variations  
✅ **Consistency:** Every frame gets a 2048-dim feature vector  
✅ **GPU Acceleration:** Significantly faster with CUDA support  
✅ **Pre-trained:** No training required, leverages ImageNet knowledge  
✅ **2-opt Optimization:** Local search improves ordering quality  

---

## Limitations

⚠️ **Computational Cost:** Slower than ORB, especially without GPU  
⚠️ **Dependencies:** Requires PyTorch (~1 GB download with dependencies)  
⚠️ **Memory:** Feature matrix requires more memory than ORB descriptors  
⚠️ **Model Size:** ResNet50 model is ~100 MB  

---

## Possible Improvements

### 1. Use Larger Models
- ResNet101 or ResNet152 for more detailed features
- EfficientNet for better accuracy/speed tradeoff

### 2. Fine-tuning
- Fine-tune ResNet50 on video frame sequences
- Learn temporal relationships explicitly

### 3. Advanced Ordering
- Use A* search with learned heuristics
- Apply beam search for multiple candidate paths
- Use dynamic programming for optimal subsequences

### 4. Feature Fusion
- Combine ResNet50 with optical flow
- Merge deep features with ORB keypoints
- Multi-scale feature extraction

### 5. Temporal Models
- Use LSTM/GRU to model temporal dependencies
- Apply transformer models for sequence modeling

---

## Expected Results

Based on semantic feature matching, ResNet50 should provide:
- **Better handling** of lighting variations
- **Improved accuracy** in complex forest scenes
- **More stable** ordering compared to ORB
- **Higher similarity scores** between consecutive frames

---

## Author

**Ansari Mohammed Umair**  
Tecdia Project - Jumbled Frames Reconstruction Challenge

---

## License

MIT License
