# V3: Reversed V2 Output

## Overview
V3 is a simple experiment that reverses the frame order from V2's output to test if the video plays better in reverse.

## What V3 Does

1. **Loads V2's frame order** from `optical_refined_frames.txt`
2. **Reverses the entire sequence** (first frame becomes last, last becomes first)
3. **Saves frames in reversed order** to `reconstructed_frames_v3_reversed/`
4. **Creates reversed video** as `reconstructed_v3_reversed.mp4`

## Key Details

- **Input**: V2's frame order (300 frames, starting with frame 276)
- **Output**: Reversed sequence (300 frames, starting with frame 48)
- **Duration**: 10.00 seconds at 30 fps
- **No ML processing**: Just simple frame order reversal

## Purpose

Since V2's video showed the person walking backward, V3 tests if simply reversing the frame order makes the person walk forward correctly.

## Files

- `reverse_video.py` - Reverses V2's frame order
- `reconstruct_video.py` - Creates video from reversed frames
- Output: `output/reconstructed_v3_reversed.mp4`

## Comparison

- **V2**: Person walks backward (frame order: 276 → ... → 48)
- **V3**: Person walks forward? (frame order: 48 → ... → 276)

## Note

This is a diagnostic approach to verify if the issue in V2 was simply the frame order direction, not the ordering logic itself.
