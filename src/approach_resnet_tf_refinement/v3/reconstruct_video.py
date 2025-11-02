import os
import cv2

def reconstruct_video_from_frames(frames_dir, output_video_path, fps=30):
    """
    Reconstruct video from ordered frames.
    
    Args:
        frames_dir: Directory containing ordered frames (frame_0000.jpg, frame_0001.jpg, etc.)
        output_video_path: Path to save the reconstructed video
        fps: Frames per second (default: 30)
    """
    print(f"\n🎬 Reconstructing video from frames...")
    print(f"   Input: {frames_dir}")
    print(f"   Output: {output_video_path}")
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        print(f"❌ No frames found in {frames_dir}")
        return
    
    print(f"   Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"❌ Could not read first frame: {first_frame_path}")
        return
    
    height, width, _ = first_frame.shape
    print(f"   Video dimensions: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"   ⚠️  Could not read {frame_file}, skipping")
            continue
        
        out.write(frame)
    
    out.release()
    
    duration = len(frame_files) / fps
    print(f"\n✅ Video reconstructed successfully!")
    print(f"   Duration: {duration:.2f} seconds ({len(frame_files)} frames)")
    print(f"   Saved to: {output_video_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # V3 reversed frames directory
    frames_dir = os.path.join(project_root, "output", "reconstructed_frames_v3_reversed")
    
    # Output video path - Final unjumbled video
    output_video = os.path.join(project_root, "output", "unjumbled_video.mp4")
    
    # Create video
    reconstruct_video_from_frames(frames_dir, output_video, fps=30)


if __name__ == "__main__":
    main()
