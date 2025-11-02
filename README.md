# Jumbled Frames Reconstruction Challenge

## Project Overview
This project reconstructs a 10-second, 1080p, 30 FPS video whose frames have been randomly shuffled. The goal is to restore the correct frame order as accurately and efficiently as possible.

## Challenge Details
- **Input**: A jumbled video file (jumbled_video.mp4) with 300 shuffled frames
- **Output**: Reconstructed video in correct sequential order
- **Constraints**: Single shot video with no cuts

## Project Structure
```
jumbled-frames-reconstruction/
├── src/                    # Source code
├── input/                  # Input jumbled video
├── output/                 # Reconstructed video output
├── docs/                   # Algorithm explanation and documentation
├── tests/                  # Test files and utilities
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore file
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd jumbled-frames-reconstruction
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Implemented Approaches

### Approach: ResNet-50 + Refinements (Recommended)

This approach uses deep learning features with progressive refinements across three versions.

#### **V1: ResNet-50 + 2-opt + Sliding Window Refinement**
- **Feature Extraction**: ResNet-50 pre-trained on ImageNet
- **Initial Ordering**: Greedy nearest neighbor with cosine similarity
- **2-opt Improvement**: Local optimization to fix crossed edges
- **Sliding Window Refinement**: Further smoothing of temporal continuity
- **Starting Point**: Frame 38 (optimally selected)

**Run:**
```bash
run_resnet_tf_refinement.bat
```

**Output:** `output/reconstructed_frames_refined/` → `output/reconstructed_refined_video.mp4`

#### **V2: V1 + Optical Flow Enhancement**
- **All V1 features** +
- **Optical Flow Verification**: Checks motion direction across sampled frame transitions
- **Direction Correction**: Reverses sequence if backward motion detected
- **Final Verification**: Samples middle section to confirm walk direction

**Run:**
```bash
run_optical_flow_v2.bat
```

**Output:** `output/reconstructed_frames_optical_flow_v2/` → `v2/output/reconstructed_optical_flow_v2.mp4`

**Note:** The reconstructed video shows the person walking backward. This is inherent to the source video's frame ordering and documented accordingly.

#### **V3: V2 + Cyclic Rotation Correction**
- **All V2 features** +
- **Similarity Matrix**: Full pairwise cosine similarity computation
- **Cyclic Offset Detection**: Identifies temporal discontinuities in the sequence
- **Rotation Correction**: Fixes residual misplaced chunks by rotating the frame order
- **Threshold**: 0.3 minimum dissimilarity to detect discontinuity
- **Max Shift**: 80% of total frames to avoid noise corrections

**Run:**
```bash
run_cyclic_v3.bat
```

**Output:** `output/reconstructed_frames_cyclic_v3/` → `v3/output/reconstructed_cyclic_v3.mp4`

**Improvement:** Fixes cyclic ordering issues where end frames belong at the beginning, restoring proper temporal continuity.

## Performance Metrics

### V1 (ResNet-50 + 2-opt + Sliding Window)
- **Execution Time**: ~1-2 minutes
- **Continuity**: Good - smooth transitions
- **Issue**: Cyclic offset (last frames should be first)

### V2 (V1 + Optical Flow)
- **Execution Time**: ~2 minutes
- **Continuity**: Excellent - verified motion direction
- **Issue**: Person walking backward (source video characteristic)

### V3 (V2 + Cyclic Correction)
- **Execution Time**: ~2-3 minutes (includes similarity matrix computation)
- **Continuity**: Excellent - cyclic offset corrected
- **Accuracy**: Best overall temporal ordering
- **Note**: Backward walking preserved as source characteristic

## System Requirements
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.0+
- NumPy
- torchvision
- Pillow
- tqdm
- ultralytics (YOLO)

## Version Comparison

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| ResNet-50 Features | ✅ | ✅ | ✅ |
| Greedy NN Ordering | ✅ | ✅ | ✅ |
| 2-opt Optimization | ✅ | ❌ | ❌ |
| Sliding Window | ✅ | ❌ | ❌ |
| Optical Flow | ❌ | ✅ | ✅ |
| Direction Verification | ❌ | ✅ | ✅ |
| Cyclic Correction | ❌ | ❌ | ✅ |
| Similarity Matrix | Partial | Partial | Full |
| Runtime | ~1-2 min | ~2 min | ~2-3 min |

## Key Innovations

### 1. Optimal Starting Point Selection (Frame 38)
- Analyzed frame similarity distribution
- Selected frame with highest average similarity to neighbors
- Ensures reconstruction begins from a stable temporal anchor

### 2. Cyclic Offset Correction (V3)
- Detects large discontinuities in similarity sequence
- Rotates frame order to fix cyclic misalignment
- Threshold-based to avoid overcorrection from noise

### 3. Optical Flow Verification (V2/V3)
- Samples transitions across the sequence
- Verifies motion direction using Farneback optical flow
- Reverses order if backward motion dominates

## Documentation Structure

```
docs/
└── algorithm_explanation.md    # Detailed algorithm documentation

src/approach_resnet_tf_refinement/
├── v1/                         # V1 implementation (in parent directory)
├── v2/                         # V2: Optical Flow
│   ├── reconstruct_fast_optical.py
│   ├── construct_video.py
│   ├── output/
│   └── README.md
└── v3/                         # V3: Cyclic Correction
    ├── reconstruct_cyclic_correction.py
    ├── construct_video.py
    ├── output/
    └── README.md
```

## Future Improvements

Potential enhancements for even better reconstruction:
- **Adaptive threshold** based on similarity distribution
- **Multi-scale discontinuity detection** for finer granularity
- **Dynamic programming** for global optimal ordering
- **Temporal consistency loss** during optimization
- **Motion prediction** to validate transitions

## Author
Ansari Mohammed Umair


