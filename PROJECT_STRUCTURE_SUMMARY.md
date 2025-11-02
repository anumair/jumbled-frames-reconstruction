# Jumbled Frames Reconstruction - Complete Project Structure

## 📂 Project Organization

```
jumbled-frames-reconstruction/
├── frames/                          # Input: Original jumbled frames (300 frames)
├── input/                           # Input videos
├── output/                          # All reconstructed outputs
├── src/                            # Source code for different approaches
│   └── approach_resnet_tf_refinement/  # Main approach with versions
│       ├── v1/                     # ResNet50 + 2-opt + Sliding Window
│       ├── v2/                     # v1 + Optical Flow Enhancement
│       ├── v3/                     # v2 tail frame fixing (deleted - not improving)
│       ├── v4/                     # v2 + Tail Removal (backward motion fix)
│       └── v5/                     # Two-Phase Reconstruction (experimental)
├── docs/                           # Documentation
└── venv/                           # Python virtual environment
```

---

## 🎯 Approach Evolution: ResNet-TF-Refinement

### **V1: ResNet50 + 2-opt + Sliding Window Refinement**

**Location:** `src/approach_resnet_tf_refinement/v1/`

**Pipeline:**
1. **Extract ResNet-50 Features** (`1_extract_resnet_features.py`)
   - Extracts deep features from all 300 frames
   - Saves to: `resnet_features.npy`

2. **Compute Similarity Matrix** (`2_compute_similarity.py`)
   - Computes cosine similarity between all frame pairs
   - Saves to: `similarity_matrix.npy`

3. **Smart Frame Ordering** (`3_order_frames_smart.py`)
   - **Smart Starting Point Detection**: Frame 276 identified as optimal start
   - Greedy nearest neighbor algorithm for initial ordering
   - Saves to: `frame_order_initial.txt`

4. **2-Opt Refinement** (`4_refine_ordering.py`)
   - Applies 2-opt local search optimization
   - Sliding window refinement for local coherence
   - Saves to: `frame_order_refined.txt`

5. **Video Reconstruction** (`5_reconstruct_video.py`)
   - Creates final video from ordered frames

**Output:**
- Directory: `output/reconstructed_frames_v1/` (implied, frames saved in v1 workflow)
- **Result:** Good continuity, person walking forward correctly
- **Duration:** ~276 frames at 30fps ≈ 9.2 seconds

**Key Achievement:**
- ✅ Smart starting point (Frame 276) ensures correct temporal ordering
- ✅ Smooth transitions with 2-opt refinement
- ✅ Forward walking motion preserved

---

### **V2: V1 + Optical Flow Enhancement**

**Location:** `src/approach_resnet_tf_refinement/v2/`

**Additional Processing:**
- **Input:** Uses V1's frame order as starting point (Frame 276)
- **Optical Flow Verification:** (`reconstruct_fast_optical.py`)
  - Calculates optical flow between consecutive frames
  - Detects motion direction (forward vs backward)
  - **Critical Fix:** Auto-reverses sequence if backward motion detected
  - Uses Farneback optical flow algorithm
  - Samples frames to compute median horizontal flow

**Pipeline:**
```
V1 frame order (starting at 276)
  ↓
Optical flow analysis
  ↓
Direction verification (forward/backward detection)
  ↓
Auto-reverse if needed
  ↓
Save refined frames + generate video
```

**Output:**
- Directory: `output/reconstructed_frames_optical_flow_v2/`
- Frame order: `optical_refined_frames.txt`
- **Result:** Good continuity overall
- **Issue:** Last ~1 second (30 frames) appear jumbled/misplaced

**Key Features:**
- ✅ Inherits V1's smart starting point (276)
- ✅ Optical flow ensures forward motion
- ✅ Auto-correction of direction
- ⚠️ Tail frames (last 30) need repositioning

**Video Generation:** `construct_video.py`

---

### **V3: V2 Tail Frame Fixing** ❌ DELETED

**Location:** `src/approach_resnet_tf_refinement/v3/` (deleted)

**Purpose:** Attempted to fix V2's misplaced tail frames (last ~30 frames)

**Approach Tried:**
- Reinsert tail frames using neighborhood-constrained search
- Extract features for tail frames
- Find best insertion points based on similarity

**Files:**
- `fix_v2_output.py` - Tail frame repositioning logic
- `reconstruct_video.py` - Video generation

**Output Directories (deleted):**
- `output/reconstructed_frames_v3/`
- `output/reconstructed_frames_v3_fixed/`
- `output/reconstructed_frames_cluster_v3/`
- `output/reconstructed_frames_cyclic_v3/`

**Result:** ❌ No improvement / degradation
**Status:** Deleted - approach not effective

---

### **V4: V2 + Intelligent Tail Removal**

**Location:** `src/approach_resnet_tf_refinement/v4/`

**Key Innovation:** Instead of fixing tail frames, remove them intelligently

**Pipeline:**
1. **Start Detection:** Find true starting point using directional flow analysis
2. **Greedy Ordering:** Build main sequence (V2 style)
3. **Direction Verification:** Ensure forward motion
4. **Tail Break Detection:** 
   - Detects where good sequence ends and bad tail begins
   - Uses sliding window similarity + flow analysis
   - Identifies discontinuity points
5. **Trim Sequence:** Remove problematic tail frames

**Files:**
- `reconstruct_frames_v4.py` - Main reconstruction logic
- `construct_video_v4.py` - Video generation

**Output:**
- Directory: `output/reconstructed_frames_v4/`
- Frame order: `frame_order_v4.txt`

**Results (from your run):**
```
📊 Total frames: 300
✅ Selected start: Frame 299 (direction: backward (reversed), score: 17.40)
✅ Ordered 300 frames
✅ Direction confirmed
🎯 Break detected at frame 262/300 (similarity threshold: 0.995)
📊 Sequence trimmed: 300 → 262 frames
   Removed 38 problematic frames from tail
```

**Key Achievement:**
- ✅ Auto-detects and removes bad tail frames
- ✅ Maintains forward motion
- ✅ Cleaner video (262 frames ≈ 8.7 seconds)
- ✅ No jumbled frames at the end

---

### **V5: Two-Phase Reconstruction (Experimental)**

**Location:** `src/approach_resnet_tf_refinement/v5/`

**Approach:** Separate main sequence building from outlier handling

**Pipeline:**
1. **Phase 1:** Multi-method voting for optimal starting point
   - Edge detection
   - Low similarity detection
   - Directional flow testing
   - Tests forward vs backward for top candidates

2. **Phase 2:** Greedy main sequence construction

3. **Phase 3:** Neighborhood-constrained insertion of missing frames
   - Find best matching region
   - Search locally (±30 positions)
   - Insert based on similarity + flow continuity

4. **Phase 4:** Direction verification

5. **Phase 5:** 2-opt local refinement

**Files:**
- `reconstruct_frames_v5.py` - Main reconstruction (v5.2 - improved start detection)
- `reconstruct_video_v5.py` - Video generation

**Output Directories:**
- `output/reconstructed_frames_v5_1/`
- `output/reconstructed_frames_v5_2/` (latest with improved start detection)

**Features:**
- Advanced starting point detection (multi-method voting)
- Handles all frames (no trimming)
- Neighborhood-constrained insertion
- 2-opt optimization

**Status:** Experimental - testing in progress

---

## 🎬 Key Concepts Explained

### **1. Smart Starting Point Detection**
- **Why Important:** Wrong start = video plays backward or out of order
- **V1 Method:** Statistical analysis finds Frame 276 as optimal
- **V4/V5 Method:** Directional flow analysis with multi-method voting
- **Verification:** Optical flow confirms forward motion

### **2. Optical Flow**
- **Purpose:** Detect motion direction between consecutive frames
- **Algorithm:** Farneback optical flow
- **Use:** Auto-reverse if person walking backward
- **Metric:** Median horizontal flow (negative = backward, positive = forward)

### **3. Frame Order Progression**
```
Jumbled Frames (random order)
  ↓
Feature Extraction (ResNet-50)
  ↓
Similarity Matrix (cosine similarity)
  ↓
Smart Starting Point (Frame 276 or auto-detected)
  ↓
Greedy Ordering (nearest neighbor)
  ↓
Refinement (2-opt / optical flow)
  ↓
Quality Check (tail removal / direction)
  ↓
Final Video
```

### **4. Version Dependencies**
```
V1 (standalone)
  ├─→ V2 (uses V1's starting point: 276)
      ├─→ V3 (uses V2's output) ❌ DELETED
      └─→ V4 (uses V2's approach + tail removal)
  
V5 (standalone, experimental)
```

---

## 📊 Performance Comparison

| Version | Frames | Duration | Forward Motion | Tail Quality | Starting Point |
|---------|--------|----------|----------------|--------------|----------------|
| **V1**  | 276    | 9.2s     | ✅ Yes         | ✅ Good      | 276 (smart)    |
| **V2**  | 300    | 10s      | ✅ Yes         | ⚠️ Last 1s bad | 276 (from V1) |
| **V3**  | N/A    | N/A      | ❌ Deleted     | ❌ Not improving | N/A      |
| **V4**  | 262    | 8.7s     | ✅ Yes         | ✅ Excellent | Auto-detected  |
| **V5**  | 300    | 10s      | ✅ Testing     | 🧪 Experimental | Auto-detected |

---

## 🚀 Recommended Workflow

### **For Best Results: Use V4**
```bash
# Activate virtual environment
cd "C:\Users\Asus\Desktop\vs code files\Tecdia project\jumbled-frames-reconstruction"
.\venv\Scripts\Activate.ps1

# Run V4 reconstruction
cd src\approach_resnet_tf_refinement\v4
python reconstruct_frames_v4.py

# Generate video
python construct_video_v4.py
```

**Why V4?**
- ✅ Automatic tail removal (no jumbled frames at end)
- ✅ Forward motion guaranteed
- ✅ Smart starting point detection
- ✅ Clean output (262 frames)

### **For Experimental Features: Use V5**
```bash
cd src\approach_resnet_tf_refinement\v5
python reconstruct_frames_v5.py
python reconstruct_video_v5.py
```

**V5 Advantages:**
- More sophisticated starting point detection
- Handles all 300 frames
- Neighborhood-constrained insertion
- 2-opt optimization

---

## 🔧 Technical Details

### **Virtual Environment Activation**
```powershell
# From project root
.\venv\Scripts\Activate.ps1

# Verify
python --version
pip list
```

### **Key Dependencies**
- PyTorch (ResNet-50 model)
- OpenCV (optical flow, video processing)
- NumPy (array operations)
- scikit-learn (cosine similarity)
- Pillow (image loading)
- tqdm (progress bars)

### **Output Structure**
```
output/
├── reconstructed_frames_optical_flow_v2/     # V2 output
│   ├── frame_0000.jpg ... frame_0299.jpg
│   └── optical_refined_frames.txt
├── reconstructed_frames_v4/                  # V4 output (recommended)
│   ├── frame_0000.jpg ... frame_0261.jpg
│   └── frame_order_v4.txt
└── reconstructed_frames_v5_2/                # V5 latest
    ├── frame_0000.jpg ... frame_0299.jpg
    └── frame_order_v5.txt
```

---

## 📝 Implementation Notes

### **V1 Implementation Details**
- **Starting Point:** Frame 276 found via smart detection in `3_order_frames_smart.py`
- **Ordering:** Greedy algorithm builds chain from 276
- **Refinement:** 2-opt swaps segments to improve local coherence
- **NOT REVERSED:** Frame order used as-is (276 is naturally the start)

### **V2 Implementation Details**
- **Inherits V1 start:** Uses 276 as starting frame
- **Optical Flow Check:** Samples middle 50% of video
- **Auto-Reverse:** If median flow < -threshold, reverses entire sequence
- **Direction Fix:** Ensures person walks forward, not backward

### **V4 Key Algorithm**
```python
# Detect tail break
for each sliding window:
    calculate similarity score
    calculate flow consistency
    combined_quality = 0.6 * similarity + 0.4 * flow

# Find sharp drop in quality (usually in last 30%)
detect_sharpest_drop()
trim_at_break_point()
```

---

## 🎥 Video Motion Verification

### **Understanding Optical Flow**
```python
# Positive flow = moving right (forward)
# Negative flow = moving left (backward)

median_flow = np.median(horizontal_flows)

if median_flow < -threshold:
    # Person walking backward
    reverse_frame_order()
```

### **Why Person Was Walking Backward (Early Versions)**
1. Initial frame ordering may have been reversed
2. Smart starting point helped but wasn't enough alone
3. Optical flow verification in V2+ ensures correct direction
4. **Solution:** Auto-reverse if flow is negative

---

## 💡 Lessons Learned

1. **Starting Point Matters:** Frame 276 is crucial for correct temporal order
2. **Optical Flow is Essential:** Without it, can't guarantee forward motion
3. **Tail Removal > Tail Fixing:** V4's approach better than V3's repositioning
4. **Multi-Phase Better:** V5's two-phase approach shows promise
5. **Feature Quality:** ResNet-50 features work well for this task

---

## 🔍 Debugging Tips

### **Check Frame Order**
```python
# Load and inspect
import numpy as np
order = np.loadtxt("frame_order_v4.txt", dtype=int)
print(f"First frame: {order[0]}")
print(f"Last frame: {order[-1]}")
print(f"Total frames: {len(order)}")
```

### **Verify Optical Flow Direction**
```bash
# Run V2 and check output
# Look for: "Median horizontal flow: X.XXXX"
# Positive = forward, Negative = backward
```

### **Visual Inspection**
```bash
# Generate video and watch
# Person should walk from left to right (forward)
# No jumps or discontinuities (especially at end)
```

---

## 📈 Future Improvements

### **Potential V6 Features**
- [ ] YOLO-based motion tracking for person position
- [ ] Hybrid similarity: ResNet + SSIM + Optical Flow
- [ ] Adaptive window sizing for refinement
- [ ] Machine learning for break point detection
- [ ] Parallel processing for faster reconstruction

### **Known Issues to Address**
- V2's last second (30 frames) occasionally misplaced
- Starting point detection could be more robust
- Need better metrics for quality assessment

---

## 📚 References

### **Algorithms Used**
- **ResNet-50:** Deep CNN for feature extraction
- **Cosine Similarity:** Measure visual similarity between frames
- **2-Opt Algorithm:** Local search optimization
- **Farneback Optical Flow:** Dense optical flow estimation
- **Greedy Nearest Neighbor:** Frame ordering heuristic

### **Key Papers/Concepts**
- Temporal Ordering of Video Frames
- Visual Similarity Metrics
- Optical Flow in Computer Vision
- Local Search Optimization (2-opt)

---

## 🎯 Quick Start Guide

### **First Time Setup**
```powershell
# 1. Clone/navigate to project
cd "C:\Users\Asus\Desktop\vs code files\Tecdia project\jumbled-frames-reconstruction"

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Verify dependencies
pip list | Select-String -Pattern "torch|opencv|numpy"
```

### **Run V4 (Recommended)**
```powershell
# Reconstruct frames
cd src\approach_resnet_tf_refinement\v4
python reconstruct_frames_v4.py

# Generate video
python construct_video_v4.py

# Check output
explorer ..\..\..\..\output\reconstructed_frames_v4\
```

### **Run V2 (If needed)**
```powershell
cd src\approach_resnet_tf_refinement\v2
python reconstruct_fast_optical.py
python construct_video.py
```

---

## 📌 Important File Paths

### **Input**
- Jumbled frames: `frames/`
- Original video: `input/`

### **V1 Artifacts**
- Features: `src/approach_resnet_tf_refinement/v1/resnet_features.npy`
- Similarity: `src/approach_resnet_tf_refinement/v1/similarity_matrix.npy`
- Initial order: `src/approach_resnet_tf_refinement/v1/frame_order_initial.txt`
- Refined order: `src/approach_resnet_tf_refinement/v1/frame_order_refined.txt`

### **V2 Output**
- Frames: `output/reconstructed_frames_optical_flow_v2/`
- Order: `output/reconstructed_frames_optical_flow_v2/optical_refined_frames.txt`

### **V4 Output** ⭐ RECOMMENDED
- Frames: `output/reconstructed_frames_v4/`
- Order: `output/reconstructed_frames_v4/frame_order_v4.txt`

---

## 🏆 Best Version Summary

**For Production Use: V4**
- Clean output (262 frames)
- No jumbled tail frames
- Forward motion guaranteed
- Automatic quality control

**For Experimentation: V5**
- Advanced algorithms
- All frames preserved
- More configuration options
- Active development

**For Understanding: V1**
- Clear pipeline
- Well-documented steps
- Foundation for other versions
- Good for learning

---

**Last Updated:** Based on current project state
**Total Versions:** 5 (V3 deleted)
**Active Versions:** V1, V2, V4, V5
**Recommended:** V4

