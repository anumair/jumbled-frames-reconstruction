# ORB-Based Frame Reconstruction Approach

## Overview

This approach reconstructs a video whose frames have been randomly shuffled using **ORB (Oriented FAST and Rotated BRIEF)** feature matching. The task involves restoring the correct chronological order so that the motion (a person walking in the forest) appears natural and continuous.

---

## Methodology

### Stage 1: Frame Extraction

**File:** `extract_frames.py`

The first stage extracts all individual frames from the jumbled video.

**Process:**
- Use OpenCV (`cv2.VideoCapture`) to read the video frame by frame
- Each frame is saved as an image (`frame_0000.jpg`, `frame_0001.jpg`, etc.) into the `frames/` directory
- For a 10-second video at 30 FPS, this results in **300 frames**

**Usage:**
```bash
python src/approach_orb/extract_frames.py
```

**Output:**
- 300 frames in `frames/` folder (~306 MB)

---

### Stage 2: Similarity Computation (Using ORB Features)

**File:** `compute_similarity.py`

To determine the correct frame order, we measure how visually similar each frame is to every other frame.

**Process:**
1. **For each frame:**
   - Convert to grayscale
   - Detect keypoints and compute descriptors using ORB (`cv2.ORB_create(nfeatures=1000)`)
   
2. **Pairwise Matching:**
   - Use Brute-Force Matcher with Hamming distance
   - Match ORB descriptors between all frame pairs
   - Similarity score based on average distance of matches: `similarity = 1 / (1 + avg_distance)`
   
3. **Store Results:**
   - Create a symmetric similarity matrix of size 300×300
   - Save as `similarity_matrix.npy` for reuse

**Key Insight:** 
Adjacent frames in time tend to have **higher similarity scores** due to visual continuity.

**Usage:**
```bash
python src/approach_orb/compute_similarity.py
```

**Output:**
- `similarity_matrix.npy` (~352 KB)

---

### Stage 3: Frame Ordering (Heuristic Greedy Search)

**File:** `order_frames_heuristic.py`

With the similarity matrix computed, we reconstruct the most probable frame order.

**Problem Formulation:**
- Path reconstruction problem (similar to TSP - Traveling Salesman Problem)
- Goal: Find a path through all frames maximizing the sum of similarities between consecutive frames

**Algorithm:**

#### 3.1 Heuristic Greedy Ordering
```
Parameters:
- alpha (0.8): Weight for immediate similarity
- beta (0.2): Weight for smoothness with recent frames
- k (5): Number of top candidates to consider

Steps:
1. Start from frame with highest average similarity (most connected)
2. At each step:
   - Consider top-k most similar unvisited frames
   - Calculate score = alpha × immediate_similarity + beta × smoothness_with_recent_frames
   - Pick frame with highest score
3. Continue until all frames are ordered
```

#### 3.2 Local Refinement
After initial ordering, apply local optimization:
- Divide sequence into overlapping windows (size 10)
- Try pairwise swaps within each window
- Keep swaps that improve local similarity continuity
- Result: 34 improvements made in our test

**Usage:**
```bash
python src/approach_orb/order_frames_heuristic.py
```

**Output:**
- `frame_order_heuristic.txt` (300 frame indices)
- `output/reconstructed_orb.mp4` (~62 MB)

---

### Stage 4: Video Reconstruction

**Integrated in:** `order_frames_heuristic.py`

Once the frame order is obtained:
1. Read frames in the computed sequence
2. Write to video using OpenCV VideoWriter at 30 FPS
3. Output saved as `reconstructed_orb.mp4`

---

## Design Considerations

| Aspect | Implementation Detail |
|--------|----------------------|
| **Accuracy** | Relies on ORB feature matching; works well for scenes with consistent textures, lighting, and camera motion |
| **Performance** | Pairwise matching optimized with caching; can be parallelized with multiprocessing |
| **Time Complexity** | O(N²) for similarity computation; acceptable for small videos (≤ 300 frames) |
| **Space Complexity** | O(N²) for similarity matrix storage |
| **Parallelism** | Matching between frame pairs can be distributed across CPU cores for speedup |
| **Storage** | Similarity matrix stored in `.npy` format for efficient reuse |

---

## Test Results

### Execution Summary

| Stage | Status | Output | Notes |
|-------|--------|--------|-------|
| Frame Extraction | ✅ PASSED | 300 frames (306 MB) | All frames extracted successfully |
| Similarity Computation | ✅ PASSED | 352 KB matrix | 90,000 pairwise comparisons |
| Frame Ordering | ✅ PASSED | 300 frame order | Starting frame: 240 |
| Local Refinement | ✅ PASSED | 34 improvements | Smoothness optimization |
| Video Reconstruction | ✅ PASSED | 62.36 MB video | Reconstructed successfully |

---

## Advantages

✅ **Lightweight and Fast:** ORB is efficient compared to SIFT/SURF  
✅ **No ML Training Required:** Purely feature-based, unsupervised approach  
✅ **Explainable Results:** Matches can be visualized to interpret ordering  
✅ **Reusable Components:** Similarity matrix can be saved and reused  
✅ **Local Optimization:** Refinement step improves ordering quality  

---

## Limitations

⚠️ **Lighting Sensitivity:** May struggle with dramatic lighting changes or motion blur  
⚠️ **Repetitive Scenes:** Less effective in background-only or highly repetitive areas  
⚠️ **Feature Quality:** ORB features are less robust than deep-learning-based embeddings for subtle differences  
⚠️ **Greedy Approximation:** Heuristic ordering doesn't guarantee global optimum  

---

## Possible Improvements

### 1. Deep Feature Embeddings
Replace ORB descriptors with CNN-based features (ResNet, VGG, or CLIP) to capture semantic continuity rather than pixel-level similarity.

### 2. Optical Flow Analysis
Use Lucas-Kanade or Farneback optical flow to compute motion vectors, providing direct temporal ordering clues.

### 3. Hybrid Similarity Metrics
Combine ORB similarity with:
- Color histogram correlation
- SSIM (Structural Similarity Index)
- Perceptual hashing

### 4. Advanced Ordering Algorithms
Replace greedy search with:
- Simulated Annealing
- Genetic Algorithms
- 2-opt optimization
- A* search with better heuristics

### 5. Temporal Consistency Heuristics
- Post-process ordering using smoothing filters
- Detect and correct temporal jumps
- Use motion coherence metrics

---

## File Structure

```
src/approach_orb/
├── extract_frames.py              # Stage 1: Frame extraction
├── compute_similarity.py          # Stage 2: ORB similarity computation
├── order_frames_heuristic.py      # Stage 3 & 4: Ordering + reconstruction
├── similarity_matrix.npy          # Cached similarity matrix
└── frame_order_heuristic.txt      # Output frame order
```

---

## How to Run

### Complete Pipeline (All Stages):

```bash
# Stage 1: Extract frames
python src/approach_orb/extract_frames.py

# Stage 2: Compute similarity matrix
python src/approach_orb/compute_similarity.py

# Stage 3 & 4: Order frames and reconstruct video
python src/approach_orb/order_frames_heuristic.py
```

### Quick Run (If similarity matrix exists):

```bash
# Just run ordering and reconstruction
python src/approach_orb/order_frames_heuristic.py
```

---

## Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
tqdm>=4.65.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Output Files

- **Frames Folder:** `frames/` (300 JPG images, ~306 MB)
- **Similarity Matrix:** `src/approach_orb/similarity_matrix.npy` (~352 KB)
- **Frame Order:** `src/approach_orb/frame_order_heuristic.txt` (300 lines)
- **Reconstructed Video:** `output/reconstructed_orb.mp4` (~62 MB, 10 sec @ 30 FPS)

---

## Performance Metrics

- **Frame Extraction Time:** ~2-5 seconds
- **Similarity Computation Time:** ~5-10 minutes (depends on CPU)
- **Frame Ordering Time:** ~10-30 seconds
- **Video Reconstruction Time:** ~5-10 seconds
- **Total Pipeline Time:** ~6-12 minutes

---

## Author

**Ansari Mohammed Umair**  
Tecdia Project - Jumbled Frames Reconstruction Challenge

---

## License

MIT License
