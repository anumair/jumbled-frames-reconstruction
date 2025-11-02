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

## Quick Start - Run Complete Pipeline

**Recommended:** Run the entire pipeline (V1 → V2 → V3) with a single command:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run complete pipeline
cd src\approach_resnet_tf_refinement
python pipeline.py
```

This will automatically:
1. Extract ResNet-50 features (V1)
2. Compute similarity matrix (V1)
3. Find smart starting point and order frames (V1)
4. Apply 2-opt + sliding window refinement (V1)
5. Enhance with optical flow (V2)
6. Create final unjumbled video (V3)

**Output:** `output/unjumbled_video.mp4` (the final reconstructed video)

---

## Implemented Approaches

### Approach: ResNet-50 + Refinements (Recommended)

This approach uses deep learning features with progressive refinements across three versions.

#### **V1: ResNet-50 + 2-opt + Sliding Window Refinement**
- **Feature Extraction**: PyTorch ResNet-50 pre-trained on ImageNet (2048-dimensional features)
- **Smart Starting Point**: Frame 276 (identified using multi-heuristic voting system)
- **Initial Ordering**: Greedy nearest neighbor with cosine similarity
- **2-opt Improvement**: Local optimization to fix crossed edges
- **Sliding Window Refinement**: Further smoothing of temporal continuity (window size 4-5)

**Run:**
```bash
cd src\approach_resnet_tf_refinement\v1
python 1_extract_resnet_features.py
python 2_compute_similarity.py
python 3_order_frames_smart.py
python 4_refine_ordering.py
python 5_reconstruct_video.py
```

**Output:** `output/reconstructed_frames_refined/reconstructed_refined_video.mp4`

**Key Features:**
- Iterative 2-opt refinement until convergence
- Sliding window optimization with exhaustive search
- Total execution time: ~2-3 minutes

#### **V2: V1 + Optical Flow Enhancement**
- **Starting Point**: Uses frame 276 from V1's smart starting point detection
- **Optical Flow Verification**: Farneback optical flow to check motion consistency
- **Direction Detection**: Samples middle 50% of sequence to verify walk direction
- **Adaptive Thresholding**: Uses MAD (Median Absolute Deviation) for robust detection
- **Direction Correction**: Reverses sequence if backward motion detected

**Run:**
```bash
cd src\approach_resnet_tf_refinement\v2
python reconstruct_fast_optical.py
python construct_video.py
```

**Output:** `output/reconstructed_frames_optical_flow_v2/reconstructed_optical_flow_v2.mp4`

**Improvements over V1:**
- Better temporal continuity with optical flow consistency
- Motion-aware ordering
- Reduced visual artifacts at frame transitions

**Known Limitation:** The video shows the person walking backward. This is due to the temporal ordering algorithm optimizing for visual similarity and flow consistency without accounting for semantic motion direction.

#### **V3: Reversed V2 Output (Diagnostic)**
- **Simple reversal**: Takes V2's frame order and reverses it completely
- **Purpose**: Test if reversing the sequence fixes the backward motion
- **No ML processing**: Just frame order reversal

**Run:**
```bash
cd src\approach_resnet_tf_refinement\v3
python reverse_video.py
python reconstruct_video.py
```

**Output:** `output/reconstructed_v3_reversed.mp4`

**Result:** V3 confirms the issue is frame order direction, showing the person walking forward when V2 is reversed.

## Performance Metrics

### V1 (ResNet-50 + 2-opt + Sliding Window)
- **Execution Time**: ~2-3 minutes
  - Feature extraction: ~30-60 seconds
  - Initial ordering: ~1 second
  - 2-opt refinement: ~30-120 seconds
  - Video reconstruction: ~30 seconds
- **Continuity**: Excellent - smooth transitions with refinement
- **Starting Point**: Frame 276 (smart selection)
- **Total Frames**: 300 frames = 10 seconds at 30 fps

### V2 (V1 + Optical Flow)
- **Execution Time**: ~2-4 minutes
  - All V1 processing +
  - Optical flow computation: ~1-2 minutes
- **Continuity**: Excellent - verified motion consistency
- **Motion Detection**: Uses Farneback optical flow
- **Issue**: Person walking backward (inherent to ordering algorithm)

### V3 (Reversed V2)
- **Execution Time**: <1 minute (just frame reordering)
- **Continuity**: Same as V2 (reversed)
- **Motion Direction**: Person walks forward (V2 reversed)
- **Purpose**: Diagnostic to confirm direction issue

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
| ResNet-50 Features | ✅ | ✅ | ❌ |
| Smart Starting Point (276) | ✅ | ✅ | N/A |
| Greedy NN Ordering | ✅ | ✅ | ❌ |
| 2-opt Optimization | ✅ | ❌ | ❌ |
| Sliding Window | ✅ | ❌ | ❌ |
| Optical Flow | ❌ | ✅ | ❌ |
| Direction Verification | ❌ | ✅ | N/A |
| Frame Reversal | ❌ | ❌ | ✅ |
| Motion Direction | Unknown | Backward | Forward |
| Runtime | ~2-3 min | ~3-4 min | <1 min |
| Best For | Complete reconstruction | Motion consistency | Quick direction fix |

## Key Innovations

### 1. Smart Starting Point Selection (V1) - Frame 276
- Multi-heuristic voting system combining:
  - Edge detection (asymmetric similarity distribution)
  - Low average similarity frames
  - Directional flow analysis
- Tests both forward and backward directions for each candidate
- Scores based on flow consistency and magnitude
- Ensures reconstruction begins from optimal temporal anchor

### 2. Two-Stage Refinement (V1)
- **2-opt Algorithm**: Iteratively swaps segments to improve local ordering
- **Sliding Window Optimization**: Exhaustive search within small windows (4-5 frames)
- Continues until convergence (no further improvements found)

### 3. Optical Flow Verification (V2)
- Uses Farneback optical flow to measure pixel-level motion
- Samples middle 50% of sequence for robust direction detection
- Adaptive thresholding using MAD (Median Absolute Deviation)
- Verifies temporal consistency across frame transitions

### 4. Simple Reversal Diagnostic (V3)
- Quick test to confirm direction issue
- No machine learning overhead
- Demonstrates that V2's ordering is correct but reversed

## Documentation Structure

```
docs/
└── algorithm_explanation.md    # Detailed algorithm documentation

src/approach_resnet_tf_refinement/
├── v1/                         # V1: ResNet-50 + 2-opt + Sliding Window
│   ├── 1_extract_resnet_features.py
│   ├── 2_compute_similarity.py
│   ├── 3_order_frames_smart.py
│   ├── 4_refine_ordering.py
│   ├── 5_reconstruct_video.py
│   └── README.md
├── v2/                         # V2: V1 + Optical Flow
│   ├── reconstruct_fast_optical.py
│   ├── construct_video.py
│   └── README.md
└── v3/                         # V3: Reversed V2 (Diagnostic)
    ├── reverse_video.py
    ├── reconstruct_video.py
    └── README.md
```

## Future Improvements

Potential enhancements for even better reconstruction:
- **Pose estimation** to determine semantic motion direction
- **Scene flow** for 3D motion understanding
- **Bidirectional ordering comparison** to detect reversed sequences
- **Temporal consistency loss** during optimization
- **Motion prediction models** to validate transitions
- **Hybrid approach** combining similarity, optical flow, AND direction detection

## Author
Ansari Mohammed Umair


