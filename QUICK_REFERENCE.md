# Quick Reference Guide - Jumbled Frames Reconstruction

## 🚀 Quick Start

### Activate Virtual Environment
```bash
# Windows
cd "C:\Users\Asus\Desktop\vs code files\Tecdia project\jumbled-frames-reconstruction"
venv\Scripts\activate
```

---

## 📂 Project Layout

```
jumbled-frames-reconstruction/
├── frames/                     # Input: 300 jumbled frames
├── output/                     # All reconstructed videos & frames
├── src/
│   └── approach_resnet_tf_refinement/
│       ├── v1/                # Baseline (Frame 276 start, 2-opt)
│       ├── v2/                # V1 + Optical Flow (best continuity)
│       ├── v3/                # Tail fixing attempts (deleted/testing)
│       ├── v4/                # Intelligent trimming (262 frames)
│       └── v5/                # Two-phase (300 frames, latest)
└── [Documentation files]
```

---

## 🎯 Which Version to Use?

| Use Case | Version | Why |
|----------|---------|-----|
| **Best Continuity** | V2 | Smoothest transitions, optical flow verified |
| **Clean & Safe** | V4 | Trimmed to 262 frames, no jumps |
| **Full Duration** | V5 | All 300 frames, sophisticated insertion |
| **Baseline** | V1 | Starting point, 2-opt refined |

---

## ⚡ Run Commands

### V1 (Baseline)
```bash
cd src/approach_resnet_tf_refinement/v1
python 1_extract_resnet_features.py
python 2_compute_similarity.py
python 3_order_frames_smart.py
python 4_refine_ordering.py
python 5_reconstruct_video.py
```

### V2 (Optical Flow - Recommended)
```bash
cd src/approach_resnet_tf_refinement/v2
python reconstruct_fast_optical.py
python construct_video.py
```

### V4 (Trimmed - Safe)
```bash
cd src/approach_resnet_tf_refinement/v4
python reconstruct_frames_v4.py
python construct_video_v4.py
```

### V5 (Latest - Full)
```bash
cd src/approach_resnet_tf_refinement/v5
python reconstruct_frames_v5.py
python reconstruct_video_v5.py
```

---

## 📊 Version Comparison

| Feature | V1 | V2 | V4 | V5 |
|---------|----|----|----|----|
| Frames | 300 | 300 | 262 | 300 |
| Duration | 10s | 10s | 8.7s | 10s |
| Starting Point | 276 (smart) | 276 (from V1) | Auto-detect | Auto-detect |
| Direction Check | ❌ | ✅ Optical Flow | ✅ Optical Flow | ✅ Optical Flow |
| Tail Issue | ⚠️ Some | ⚠️ Last 1s | ✅ Trimmed | ❓ Testing |
| 2-opt Refinement | ✅ | ❌ | ❌ | ✅ |
| Smart Insertion | ❌ | ❌ | ❌ | ✅ |

---

## 🎬 Output Videos Location

All videos in: `output/`

**Main Approach:**
- `reconstructed_resnet_tf_refined_v1.mp4` - V1 output
- `reconstructed_optical_flow_v2.mp4` - V2 output ⭐
- `reconstructed_video_v4.mp4` - V4 output ✅
- `reconstructed_v5.mp4` - V5 output 🆕

**Other Approaches:**
- `reconstructed_orb.mp4` - ORB features
- `reconstructed_resnet50.mp4` - Basic ResNet
- `reconstructed_resnet_yolo.mp4` - ResNet + YOLO

---

## 🔧 Key Parameters

### V1/V2 Starting Point
```python
start_idx = 276  # Smart detected in V1
```

### V4 Tail Trimming
```python
tail_start = 270  # Trim after frame 270
# Keeps first 262 frames (87% of video)
```

### V5 Neighborhood Size
```python
neighborhood_size = 30  # ±30 frames search radius
```

### Optical Flow Threshold
```python
threshold = max(0.5, 1.5 * mad)  # Adaptive
# Reverses if median_flow < -threshold
```

---

## 📝 File Locations

### Frame Orders
- V1: `src/approach_resnet_tf_refinement/v1/frame_order_refined.txt`
- V2: `output/reconstructed_frames_optical_flow_v2/optical_refined_frames.txt`
- V4: `output/reconstructed_frames_v4/frame_order_v4.txt`
- V5: `output/reconstructed_frames_v5_2/frame_order_v5.txt`

### Features
- ResNet: `src/approach_resnet_tf_refinement/v1/resnet_features.npy`
- Similarity: `src/approach_resnet_tf_refinement/v1/similarity_matrix.npy`

### Reconstructed Frames
- V1: N/A (frames saved inline during video creation)
- V2: `output/reconstructed_frames_optical_flow_v2/`
- V4: `output/reconstructed_frames_v4/`
- V5: `output/reconstructed_frames_v5_2/`

---

## 🐛 Known Issues

### V2: Tail Frame Problem
- **Issue:** Last ~30 frames (1 second) appear misplaced
- **Status:** Identified but not fixed in V2
- **Workaround:** Use V4 (trimmed) or V5 (better insertion)

### V3: Deleted
- **Reason:** Multiple approaches tested, none improved results
- **Tested:** Cyclic rotation, YOLO, SSIM, outlier detection
- **Result:** All ineffective, folder kept for reference

### V4: Shorter Duration
- **Trade-off:** Trimmed to 262 frames for quality
- **Lost:** 38 frames (~1.3 seconds)
- **Benefit:** Clean, no jumps

---

## 🔬 Technical Details

### ResNet-50 Feature Extraction
```python
model = models.resnet50(weights='IMAGENET1K_V1')
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
# Output: 2048-dimensional feature vector per frame
```

### Optical Flow (Farneback)
```python
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)
horizontal_flow = np.median(flow[..., 0])  # Median is robust
```

### 2-opt Local Search
```python
for i in range(1, n-2):
    for k in range(i+1, min(i+20, n-1)):
        new_order = order[:i] + order[i:k+1][::-1] + order[k+1:]
        if cost(new_order) > cost(current_order):
            current_order = new_order
```

---

## 📚 Documentation

- `README.md` - Project overview
- `PROJECT_COMPLETE_SUMMARY.md` - Comprehensive guide (all versions)
- `PROJECT_STRUCTURE_SUMMARY.md` - Detailed structure
- `QUICK_REFERENCE.md` - This file
- `docs/algorithm_explanation.md` - Algorithm details

---

## 🌐 Git Commands

```bash
# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Your message"

# Push to GitHub
git push origin main

# Pull latest
git pull origin main

# View history
git log --oneline -10
```

---

## 🎯 Recommendations

### For Production Use:
1. **First choice:** V2 (if last 1s is acceptable)
2. **Second choice:** V4 (clean, shorter)
3. **Testing:** V5 (evaluate quality first)

### For Development:
1. Test V5 output thoroughly
2. Compare V2 vs V4 vs V5 side-by-side
3. Consider hybrid: V2's approach + V4's tail detection

---

## 📞 Quick Troubleshooting

### Virtual Environment Not Activating
```bash
# Re-create if needed
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues
```python
# Code automatically falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Video Not Playing
- Check codec: H.264 (mp4v)
- Try VLC player
- Ensure frame rate: 30 fps

---

**Last Updated:** 2025-11-02  
**Current Best:** V2 (continuity) or V4 (reliability)  
**Latest:** V5 (testing)
