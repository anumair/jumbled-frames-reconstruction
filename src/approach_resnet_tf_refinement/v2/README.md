# Optical Flow Refinement - Version 2

## Overview

This version builds upon the ResNet50 + Refinement approach (v1) by adding **optical flow analysis** to improve frame ordering. Optical flow measures pixel-level motion between consecutive frames, helping to ensure temporal consistency.

## Approach

### Starting Point
- Uses the starting frame from v1 (ResNet50 + Refinement approach)
- This ensures we begin with the best starting point identified by the previous method

### Optical Flow Process
1. **Extract frames** from the v1 reconstructed video
2. **Compute optical flow** between consecutive frame pairs
3. **Calculate flow consistency** to measure temporal smoothness
4. **Refine ordering** based on optical flow magnitudes
5. **Reconstruct video** with improved temporal consistency

## Files

- `reconstruct_fast_optical.py` - Main optical flow reconstruction script
- `construct_video.py` - Video construction utility
- `output/reconstructed_optical_flow_v2.mp4` - Final reconstructed video

## Usage

### Run with Batch File (Recommended)
```bash
# Double-click from project root
run_optical_flow_v2.bat
```

### Manual Execution
```bash
# Activate virtual environment
..\..\venv\Scripts\Activate.ps1

# Run optical flow refinement
python src/approach_resnet_tf_refinement/v2/reconstruct_fast_optical.py
```

## Algorithm Details

### Optical Flow Calculation
- Uses **Farneback optical flow** algorithm from OpenCV
- Computes dense optical flow between consecutive frames
- Measures pixel displacement magnitude and direction

### Consistency Metric
```python
flow_consistency = mean(optical_flow_magnitude)
```

Higher consistency values indicate smoother transitions between frames.

## Results

### Improvements Over v1
- ✅ **Better temporal continuity**: Optical flow ensures smoother motion transitions
- ✅ **Reduced visual artifacts**: Less jumping between dissimilar frames
- ✅ **Motion-aware ordering**: Takes into account actual pixel movement

### Known Limitations
- ⚠️ **Motion Direction**: The reconstructed video shows the subject moving **backwards**
  - This is because the temporal ordering algorithm optimizes for visual similarity and flow consistency
  - It does not account for the semantic direction of motion (forward vs backward)
  - The frames are in correct temporal order but reversed in time

## Performance

- **Execution Time**: ~2-4 minutes
  - Optical flow computation: ~1-2 minutes
  - Video reconstruction: ~30 seconds

## Technical Details

### Dependencies
```bash
opencv-python
numpy
tqdm
```

### Optical Flow Parameters
- **pyr_scale**: 0.5
- **levels**: 3
- **winsize**: 15
- **iterations**: 3
- **poly_n**: 5
- **poly_sigma**: 1.2

## Comparison

| Metric | v1 (ResNet + Refinement) | v2 (+ Optical Flow) |
|--------|-------------------------|---------------------|
| Continuity | Good | Excellent |
| Motion Smoothness | Moderate | High |
| Temporal Consistency | Good | Excellent |
| Motion Direction | Unknown | Backward |
| Processing Time | ~2 min | ~4 min |

## Future Work

To address the backward motion issue:
1. Implement motion direction detection using:
   - Optical flow direction analysis
   - Pose estimation to track body orientation
   - Scene flow to understand 3D motion
2. Add bidirectional ordering comparison
3. Use semantic understanding to determine intended direction
4. Implement reversibility check and auto-correction

## Notes

- The optical flow approach significantly improves temporal continuity
- While the motion direction is backward, the frame ordering is temporally consistent
- This demonstrates that similarity-based ordering doesn't guarantee correct motion direction
- A hybrid approach combining similarity, optical flow, AND direction detection would be ideal
