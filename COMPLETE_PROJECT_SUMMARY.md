# Jumbled Frames Reconstruction - Complete Project Summary

## Project Overview
A comprehensive video frame reconstruction system that takes jumbled/shuffled video frames and reconstructs them in the correct temporal order using deep learning and computer vision techniques.

**Dataset**: 300 jumbled frames from a 10-second video (30 fps)
**Goal**: Reconstruct frames in correct temporal order with person walking forward

---

## 📁 Project Structure

```
jumbled-frames-reconstruction/
├── frames/                          # Input: 300 jumbled frames
├── input/                          # Original video input
├── output/                         # All reconstruction outputs
│   ├── reconstructed_frames_v1/
│   ├── reconstructed_frames_optical_flow_v2/
│   ├── reconstructed_video_v3/
│   ├── reconstructed_frames_v4/
│   └── reconstructed_frames_v5_2/
├── src/
│   ├── approach_orb/              # ORB feature-based approach (abandoned)
│   ├── approach_resnet50/         # Basic ResNet50 approach
│   ├── approach_resnet_tf_refinement/  # Main approach with versions
│   │   ├── v1/                    # ResNet50 + 2-opt + Sliding Window
│   │   ├── v2/                    # v1 + Optical Flow Enhancement
│   │   ├── v3/                    # Video reconstruction from v2
│   │   ├── v4/                    # V2 + Intelligent Tail Removal (deleted)
│   │   └── v5/                    # Two-Phase Reconstruction (deleted)
│   └── approach_resnet_yolo/      # YOLO-based approach (incomplete)
├── venv/                          # Python virtual environment
├── requirements.txt               # Python dependencies
└── README.md                      # Main documentation
```

---

## 🔧 Technical Approaches Implemented

### **Approach 1: ORB Feature Matching**
- **Location**: `src/approach_orb/`
- **Method**: Used ORB (Oriented FAST and Rotated BRIEF) features
- **Status**: ❌ Abandoned - Insufficient for temporal ordering
- **Reason**: ORB features good for matching but not temporal sequencing

### **Approach 2: Basic ResNet50**
- **Location**: `src/approach_resnet50/`
- **Method**: ResNet50 deep features + greedy nearest neighbor
- **Status**: ✅ Baseline established
- **Performance**: Basic ordering but no refinement

### **Approach 3: ResNet + TensorFlow Refinement (Main Approach)**
**Location**: `src/approach_resnet_tf_refinement/`

This is our primary approach with multiple iterative versions:

#### **Version 1 (v1)** ✅ **WORKING**
**Location**: `src/approach_resnet_tf_refinement/v1/`
**Output**: `output/reconstructed_frames_v1/`

**Pipeline**:
1. **Feature Extraction** (`1_extract_features.py`)
   - ResNet50 deep features from ImageNet
   - Saves: `resnet_features.npy`

2. **Similarity Matrix** (`2_compute_similarity.py`)
   - Cosine similarity between all frame pairs
   - Saves: `similarity_matrix.npy`

3. **Smart Starting Point** (`3_order_frames_smart.py`)
   - Multi-method voting system:
     - Edge detection (asymmetric similarity)
     - Low average similarity (typical endpoints)
     - Directional flow consistency test
   - **Result**: Frame 276 identified as optimal start
   - Saves: `frame_order_initial.txt`

4. **2-opt Refinement** (`4_refine_2opt.py`)
   - Local optimization using 2-opt swaps
   - Improves local coherence
   - Saves: `frame_order_refined.txt`

5. **Sliding Window Refinement** (`5_sliding_window.py`)
   - Fine-tunes ordering in local windows
   - Saves final order: `frame_order_refined.txt`

6. **Video Construction** (`6_construct_video.py`)
   - Creates video from reconstructed frames
   - Output: `reconstructed_v1.mp4`

**Result**: ✅ Good continuity, person walks forward, ~9 seconds of correct sequence

---

#### **Version 2 (v2)** ✅ **BEST RESULT**
**Location**: `src/approach_resnet_tf_refinement/v2/`
**Output**: `output/reconstructed_frames_optical_flow_v2/`

**Improvements over v1**:
1. **Uses v1's optimized starting point** (Frame 276)
2. **Optical Flow Enhancement** (`reconstruct_fast_optical.py`)
   - Farneback optical flow for motion continuity
   - Verifies forward motion direction
   - **Adaptive reversal** if person walking backward detected
   - Median flow analysis for robustness

3. **Video Construction** (`construct_video.py`)
   - Output: `reconstructed_optical_flow_v2.mp4`

**Key Features**:
```python
# Optical flow direction verification
median_flow = np.median(horizontal_flows)
if median_flow < -threshold:
    frame_order = frame_order[::-1]  # Reverse if backward
```

**Result**: ✅✅ **BEST PERFORMANCE** - Excellent continuity, correct forward motion, ~9 seconds perfect

**⚠️ Known Issue**: Last ~1 second (30 frames) contains misplaced frames that need repositioning

---

#### **Version 3 (v3)** ✅ **VIDEO UTILITY**
**Location**: `src/approach_resnet_tf_refinement/v3/`
**Purpose**: Video construction from v2 output

**Scripts**:
1. `construct_video.py` - Creates video from v2 frames
2. `construct_video_reverse.py` - Creates reversed video (for testing)

**Output**: 
- `reconstructed_video_v3.mp4` - Standard video from v2
- `reconstructed_video_v3_reversed.mp4` - Reversed version

**Status**: ✅ Utility version for video generation
**Note**: Not a reconstruction improvement, just video creation tool

---

#### **Version 4 (v4)** ❌ **DELETED**
**Attempted**: Intelligent tail removal
**Method**: Detect sequence breaks and trim bad frames
**Result**: ❌ No improvement over v2
**Status**: Deleted due to lack of improvement

---

#### **Version 5 (v5)** ❌ **DELETED**
**Attempted**: Two-phase reconstruction with neighborhood-constrained insertion
**Method**: 
1. Build main sequence (greedy)
2. Insert missing frames using local search
3. 2-opt refinement

**Result**: ❌ Did not improve over v2
**Status**: Deleted due to lack of improvement

---

### **Approach 4: ResNet + YOLO**
**Location**: `src/approach_resnet_yolo/`
**Method**: Combined ResNet features with YOLO object detection
**Status**: ⚠️ Partially implemented, not completed
**Components**:
- YOLO for person detection
- Centroid tracking
- Motion-based refinement

**Not pursued further** as v2 already achieved excellent results

---

## 🎯 Best Performing Solution

### **Winner: approach_resnet_tf_refinement/v2**

**Why it's the best**:
1. ✅ **Smart starting point** from v1 (Frame 276)
2. ✅ **Optical flow verification** ensures forward motion
3. ✅ **Excellent temporal continuity** (~9 seconds perfect)
4. ✅ **Robust direction detection** using median flow
5. ✅ **Adaptive correction** (auto-reverses if needed)

**Performance Metrics**:
- **Total frames**: 300
- **Correctly ordered**: ~270 frames (90%)
- **Video quality**: Smooth, natural walking motion
- **Duration**: ~9 seconds of perfect reconstruction
- **Issue**: Last ~30 frames misplaced (known limitation)

**Output Location**: 
- Frames: `output/reconstructed_frames_optical_flow_v2/`
- Video: `output/reconstructed_frames_optical_flow_v2/reconstructed_optical_flow_v2.mp4`
- Frame Order: `output/reconstructed_frames_optical_flow_v2/optical_refined_frames.txt`

---

## 🔬 Technical Details

### **Core Technologies**
1. **Deep Learning**: 
   - ResNet50 (PyTorch) for feature extraction
   - Pre-trained on ImageNet

2. **Computer Vision**:
   - Optical Flow (Farneback method)
   - OpenCV for video processing

3. **Optimization**:
   - 2-opt local search
   - Sliding window refinement
   - Greedy nearest neighbor

4. **Similarity Metrics**:
   - Cosine similarity
   - Optical flow magnitude
   - Motion continuity

### **Key Algorithms**

#### **Smart Starting Point Detection** (v1)
```python
# Multi-method voting:
# 1. Edge detection via asymmetric similarity
edge_score = top_20_avg - bottom_20_avg

# 2. Low average similarity (endpoints)
avg_sims = similarity_matrix.mean(axis=1)

# 3. Flow direction testing
forward_score = flow_consistency + flow_magnitude
```

#### **Optical Flow Refinement** (v2)
```python
# Calculate flow between consecutive frames
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, ...)

# Verify direction
median_flow = np.median(horizontal_flows)
if median_flow < -threshold:
    frame_order.reverse()  # Fix backward motion
```

---

## 📊 Results Comparison

| Version | Method | Continuity | Direction | Duration | Status |
|---------|--------|-----------|-----------|----------|--------|
| v1 | ResNet+2-opt+Sliding | Good | ✅ Forward | ~9s | ✅ Working |
| v2 | v1 + Optical Flow | Excellent | ✅ Forward | ~9s | ✅✅ **BEST** |
| v3 | Video from v2 | N/A | ✅ Forward | ~9s | ✅ Utility |
| v4 | Tail Removal | No improvement | ✅ Forward | ~8.7s | ❌ Deleted |
| v5 | Two-Phase | No improvement | ✅ Forward | ~10s | ❌ Deleted |

---

## 🚀 How to Run

### **Setup**
```bash
# 1. Navigate to project
cd "C:\Users\Asus\Desktop\vs code files\Tecdia project\jumbled-frames-reconstruction"

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies (if needed)
pip install -r requirements.txt
```

### **Run V1 (Full Pipeline)**
```bash
cd src\approach_resnet_tf_refinement\v1

# Step-by-step:
python 1_extract_features.py
python 2_compute_similarity.py
python 3_order_frames_smart.py
python 4_refine_2opt.py
python 5_sliding_window.py
python 6_construct_video.py
```

### **Run V2 (Best Method)** ⭐
```bash
cd src\approach_resnet_tf_refinement\v2
python reconstruct_fast_optical.py
python construct_video.py
```

### **Run V3 (Video Generation)**
```bash
cd src\approach_resnet_tf_refinement\v3
python construct_video.py
```

---

## 📝 Key Files

### **Configuration**
- `requirements.txt` - Python dependencies
- `.gitignore` - Git exclusions

### **Documentation**
- `README.md` - Main project documentation
- `COMPLETE_PROJECT_SUMMARY.md` - This file
- `EXPLANATION_V3_VS_V4.md` - Version comparisons
- `QUICK_REFERENCE.md` - Quick command reference

### **Execution Logs**
- `execution_log_hybrid.txt` - Hybrid approach logs
- `execution_log_resnet50.txt` - ResNet50 logs

### **Data Files**
- `frame_order.txt` - Various frame orderings
- `resnet_features.npy` - Extracted features
- `similarity_matrix.npy` - Frame similarities

### **Model Weights**
- `yolo11n.pt`, `yolo11x.pt`, `yolov8n.pt` - YOLO models (for approach 4)

---

## 🎓 Lessons Learned

### **What Worked** ✅
1. **ResNet50 features** - Excellent semantic understanding
2. **Smart starting point** - Critical for correct ordering
3. **Optical flow** - Essential for direction verification
4. **Iterative refinement** - 2-opt + sliding window effective
5. **Median statistics** - More robust than mean for flow

### **What Didn't Work** ❌
1. **ORB features** - Too local, no temporal info
2. **Pure YOLO** - Overcomplicated without better results
3. **Tail removal** - Discarding frames reduced quality
4. **Over-optimization** - v5's complex insertion didn't help

### **Best Practices**
1. ✅ Start with simple baseline (ResNet)
2. ✅ Add refinements incrementally
3. ✅ Validate each step with videos
4. ✅ Use robust statistics (median > mean)
5. ✅ Know when to stop (v2 is good enough)

---

## 🔮 Future Improvements

### **Potential Enhancements**
1. **Fix last 30 frames** - Targeted repositioning algorithm
2. **LSTM/Transformer** - Learn temporal patterns directly
3. **Multi-scale features** - Combine low/mid/high-level features
4. **Clustering** - Group similar scenes before ordering
5. **3D CNNs** - Use temporal convolutions (e.g., I3D, R(2+1)D)

### **Alternative Approaches**
1. **Siamese Networks** - Learn pairwise temporal relations
2. **Graph Neural Networks** - Model frame relationships
3. **Reinforcement Learning** - Learn optimal ordering policy
4. **Contrastive Learning** - Self-supervised temporal embeddings

---

## 📦 Dependencies

### **Core Libraries**
```
torch==2.5.1
torchvision==0.20.1
opencv-python==4.10.0.84
numpy==2.1.3
scikit-learn==1.5.2
Pillow==11.0.0
tqdm==4.67.1
```

### **Optional (YOLO approach)**
```
ultralytics==8.3.63
```

---

## 🏆 Final Recommendation

**Use Version 2 (v2)** from `approach_resnet_tf_refinement` for:
- ✅ Best overall quality
- ✅ Correct forward motion
- ✅ Excellent continuity
- ✅ Simple and reliable
- ✅ ~90% accuracy (270/300 frames)

**Output**: `output/reconstructed_frames_optical_flow_v2/reconstructed_optical_flow_v2.mp4`

---

## 📄 Git Repository

The project is version-controlled with Git and includes:
- All source code
- Documentation
- Configuration files
- Execution scripts
- `.gitignore` for excluding large files (models, venv, outputs)

---

## 👥 Credits

**Project**: Jumbled Video Frame Reconstruction
**Approach**: Hybrid deep learning + computer vision
**Best Method**: ResNet50 + Optical Flow (v2)
**Status**: ✅ **Production Ready** (with known limitations)

---

**Last Updated**: January 2025
**Version**: 2.0 (Final)
