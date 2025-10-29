# ResNet50 PyTorch with Refinement Approach

## Overview

This approach uses **PyTorch ResNet50** for feature extraction followed by a **two-stage refinement process** to optimize the frame ordering. The refinement step uses local search algorithms to improve the initial greedy solution.

## Key Features

1. **PyTorch Implementation**: Uses PyTorch ResNet50 with ImageNet pre-trained weights
2. **Two-Stage Refinement**:
   - **2-opt Algorithm**: Swaps segments to find local improvements
   - **Sliding Window Optimization**: Optimizes small windows using exhaustive search
3. **Iterative Improvement**: Continuously refines ordering until convergence

## Pipeline Stages

### Stage 1: Feature Extraction
**Script**: `1_extract_resnet_features.py`

- Loads pre-trained ResNet50 from PyTorch
- Removes classification layer
- Uses ImageNet pre-trained weights
- Extracts 2048-dimensional feature vectors
- Preprocesses images to 224x224 RGB

**Output**: `resnet_features.npy` (300 × 2048)

### Stage 2: Similarity Computation
**Script**: `2_compute_similarity.py`

- Computes cosine similarity between all frame pairs
- Creates 300×300 similarity matrix
- Higher values indicate more similar frames

**Output**: `similarity_matrix.npy` (300 × 300)

### Stage 3: Initial Ordering
**Script**: `3_order_frames.py`

- Uses greedy nearest neighbor algorithm
- Starts from frame with highest total similarity
- Always selects most similar unvisited frame
- Fast but may produce sub-optimal results

**Output**: `frame_order_initial.txt`

### Stage 4: Refinement (KEY INNOVATION)
**Script**: `4_refine_ordering.py`

This is the core innovation of this approach. It applies two refinement techniques:

#### 4.1 2-opt Refinement
- Classic optimization technique from TSP (Traveling Salesman Problem)
- Iteratively tries swapping segments of the path
- Accepts swaps that improve total similarity
- Continues until no improvement found

**How it works**:
```
Original: [A, B, C, D, E, F]
Try swap: [A, B, D, C, E, F]  (reverse C-D)
If better: Keep it and continue
If worse:  Try next swap
```

#### 4.2 Sliding Window Optimization
- Moves a small window (size 4-5) across the sequence
- For each window, tries all permutations
- Selects the best local arrangement
- Ensures local optimality

**How it works**:
```
Sequence: [..., F1, F2, F3, F4, F5, ...]
         
Window 1: [F1, F2, F3, F4] → Try all 24 permutations
Window 2: [F2, F3, F4, F5] → Try all 24 permutations
...
```

**Output**: `frame_order_refined.txt`

### Stage 5: Video Reconstruction
**Script**: `5_reconstruct_video.py`

- Reads frames in refined order
- Writes video at 30 FPS
- Uses MP4 format with H264 codec

**Output**: `output/reconstructed_resnet_tf_refined.mp4`

## Usage

### Method 1: Run Complete Pipeline (Recommended)
```bash
# Double-click the batch file
run_resnet_tf_refinement.bat
```

### Method 2: Run Individual Steps
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Step 1: Extract features
python src/approach_resnet_tf_refinement/1_extract_resnet_features.py

# Step 2: Compute similarity
python src/approach_resnet_tf_refinement/2_compute_similarity.py

# Step 3: Initial ordering
python src/approach_resnet_tf_refinement/3_order_frames.py

# Step 4: Refine ordering (IMPORTANT!)
python src/approach_resnet_tf_refinement/4_refine_ordering.py

# Step 5: Reconstruct video
python src/approach_resnet_tf_refinement/5_reconstruct_video.py
```

## Dependencies

```bash
pip install torch torchvision opencv-python numpy scikit-learn tqdm pillow
```

**Note**: Uses PyTorch (same as hybrid approach)

## Algorithm Details

### Greedy Algorithm Complexity
- Time: O(n²) where n = number of frames
- Space: O(n²) for similarity matrix

### 2-opt Refinement Complexity
- Time: O(n² × k) where k = iterations (typically < 50)
- Space: O(n)

### Sliding Window Complexity
- Time: O(n × w!) where w = window size (4-5)
- Space: O(n)
- With w=4: 24 permutations per window
- With w=5: 120 permutations per window

## Advantages

1. **Better Optimization**: Refinement can escape local minima from greedy algorithm
2. **TensorFlow Ecosystem**: Easy to integrate with other TF models
3. **Proven Techniques**: 2-opt is well-established in combinatorial optimization
4. **Tunable**: Can adjust window size vs. computation time trade-off
5. **Measurable Improvement**: Reports before/after similarity scores

## Comparison with Other Approaches

| Approach | Feature Source | Optimization | Refinement |
|----------|---------------|--------------|------------|
| ORB | Handcrafted | Greedy | None |
| ResNet50 (PyTorch) | Deep Learning | Greedy | None |
| ResNet50 + YOLO | Multi-model | Greedy | None |
| **ResNet50 + Refinement** | Deep Learning | **2-opt + Window** | ✅ **Yes** |

## Expected Performance

- **Initial Ordering**: ~85-90% accuracy (similar to greedy ResNet)
- **After Refinement**: ~92-95% accuracy (5-10% improvement expected)
- **Execution Time**: 
  - Feature extraction: ~30-60 seconds
  - Initial ordering: ~1 second
  - Refinement: ~30-120 seconds (depends on iterations)
  - Total: ~2-3 minutes

## Future Enhancements

1. Add simulated annealing for global optimization
2. Implement genetic algorithm for population-based search
3. Use optical flow to validate temporal consistency
4. Add beam search for initial ordering (instead of greedy)
5. Parallel processing for sliding window optimization
6. Adaptive window size based on local similarity

## References

- 2-opt Algorithm: Lin-Kernighan heuristic for TSP
- ResNet50: He et al., "Deep Residual Learning for Image Recognition"
- Local Search: Russell & Norvig, "Artificial Intelligence: A Modern Approach"
