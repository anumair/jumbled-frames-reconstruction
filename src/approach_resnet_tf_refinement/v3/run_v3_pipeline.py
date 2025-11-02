"""
V3 Comprehensive Reconstruction Pipeline
=========================================

This pipeline implements a multi-stage approach:

1. Outlier Detection: Remove frames that don't belong to main sequence
2. Connected Components: Find largest cohesive group of frames
3. Smart Start Point: Use optical flow to find true starting frame
4. Greedy Ordering: Order frames with similarity threshold
5. Optical Flow Verification: Check and correct direction
6. Temporal Outlier Removal: Final cleanup of misplaced frames
7. Video Reconstruction: Create final video

Run this script to execute the entire v3 pipeline.
"""

import os
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from comprehensive_reconstruction import ComprehensiveFrameReconstructor
from reconstruct_video import reconstruct_video

def main():
    print("\n" + "=" * 70)
    print("V3 COMPREHENSIVE RECONSTRUCTION PIPELINE")
    print("=" * 70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "frames")
    output_frames_dir = os.path.join(project_root, "output", "reconstructed_frames_v3")
    output_video = os.path.join(project_root, "output", "reconstructed_v3.mp4")
    
    # Stage 1: Comprehensive frame reconstruction
    print("\n🚀 Stage 1: Comprehensive Frame Reconstruction")
    print("-" * 70)
    reconstructor = ComprehensiveFrameReconstructor(frames_dir, output_frames_dir)
    frame_order = reconstructor.reconstruct()
    
    # Stage 2: Video reconstruction
    print("\n\n🚀 Stage 2: Video Reconstruction")
    print("-" * 70)
    reconstruct_video(output_frames_dir, output_video, fps=30)
    
    print("\n\n" + "=" * 70)
    print("✅ V3 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Reconstructed frames: {output_frames_dir}")
    print(f"🎬 Output video: {output_video}")
    print(f"📊 Total frames processed: {len(frame_order)}")
    print("\nYou can now view the reconstructed video.")
    

if __name__ == "__main__":
    main()
