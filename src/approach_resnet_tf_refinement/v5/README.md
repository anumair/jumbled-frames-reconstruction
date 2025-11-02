# V5: Two-Phase Frame Reconstruction with Neighborhood-Constrained Insertion

## Overview
V5 implements a sophisticated two-phase reconstruction approach that addresses the "missing frames" problem seen in previous versions. Instead of discarding frames, it intelligently inserts them back into the sequence using neighborhood-constrained search.

## Key Features

### Phase 1: Optimal Starting Point Detection
- Uses directional optical flow analysis
- Tests multiple candidate start points
- Evaluates both forward and backward motion consistency
- Selects start point with best flow coherence

### Phase 2: Main Sequence Construction
- Greedy nearest-neighbor ordering from optimal start
- Builds a strong core sequence
- Identifies frames that don't fit the main flow

### Phase 3: Neighborhood-Constrained Insertion ⭐ **NEW**
- **Problem Solved**: Previous versions excluded frames that didn't fit greedy ordering
- **Solution**: Smart re-insertion of missing frames
- **Method**: 
  - Finds the best matching region for each missing frame
  - Searches locally (±30 positions) around that region
  - Inserts frame at position that maximizes:
    - Similarity to immediate neighbors
    - Flow consistency
  - Penalizes breaking strong existing connections
  - **Result**: All frames are used, maintaining temporal locality

### Phase 4: Direction Verification
- Samples middle 50% of sequence
- Uses optical flow to detect backward motion
- Reverses sequence if needed

### Phase 5: 2-Opt Local Refinement
- Fine-tunes ordering using segment swaps
- Improves local coherence
- 3 iterations for optimization

## Improvements Over Previous Versions

| Version | Approach | Frames Used | Continuity | Issue |
|---------|----------|-------------|------------|-------|
| V2 | Greedy + Optical Flow | ~270/300 | Excellent | Missing 30 frames |
| V3 | Outlier Detection | ~230/300 | Good | Too aggressive filtering |
| V4 | Tail Removal | ~262/300 | Very Good | Still missing frames |
| **V5** | **Two-Phase + Insertion** | **300/300** ✅ | **Excellent** | **All frames preserved** |

## Parameters

```python
neighborhood_size = 30  # Search radius for frame insertion
iterations = 3          # 2-opt refinement iterations
```

## Usage

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run reconstruction
python src\approach_resnet_tf_refinement\v5\reconstruct_frames_v5.py

# Create video
python src\approach_resnet_tf_refinement\v5\reconstruct_video_v5.py
```

## Output
- **Frames**: `output/reconstructed_frames_v5_1/`
- **Video**: `output/reconstructed_v5.mp4`
- **Frame Order**: `output/reconstructed_frames_v5_1/frame_order_v5.txt`

## Technical Details

### Neighborhood-Constrained Insertion Algorithm
1. **Find Best Region**: Calculate average similarity to windows across sequence
2. **Local Search**: Search ±30 positions around best region
3. **Score Calculation**:
   - Similarity to neighbors (distance 1 and 2)
   - Penalty for breaking strong connections (>0.95 similarity)
   - Bonus for maintaining smooth transitions
4. **Insert**: Place frame at position with highest score

### Why This Works
- **Maintains temporal locality**: Frames are inserted near similar content
- **Preserves existing quality**: Penalties prevent breaking good sequences
- **Handles outliers gracefully**: Even difficult frames find a reasonable position
- **Complete reconstruction**: No frames are discarded

## Expected Results
- ✅ All 300 frames included (no duplicates)
- ✅ 10-second video at 30fps
- ✅ Smooth motion continuity
- ✅ No abrupt jumps or misplaced segments

## Comparison with V2
- **V2**: Excellent smoothness but missing ~30 frames
- **V5**: Same smoothness + all frames included
- **Method**: V5 = V2's greedy core + smart insertion of V2's "excluded" frames
