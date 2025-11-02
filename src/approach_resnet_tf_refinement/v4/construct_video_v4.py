import cv2
import os
from tqdm import tqdm

def construct_video(frames_dir, output_video_path, fps=30):
    """
    Construct video from ordered frames in the directory.
    
    Args:
        frames_dir: Directory containing ordered frames (frame_0000.jpg, frame_0001.jpg, ...)
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
    """
    print("=" * 70)
    print("VIDEO CONSTRUCTION (V4)")
    print("=" * 70)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.lower().endswith(('.jpg', '.png'))])
    
    if not frame_files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    
    print(f"\n📊 Found {len(frame_files)} frames")
    print(f"📁 Input directory: {frames_dir}")
    print(f"🎬 Output video: {output_video_path}")
    print(f"⏱️  FPS: {fps}")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {first_frame_path}")
    
    height, width = first_frame.shape[:2]
    print(f"📐 Resolution: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError("Failed to create video writer")
    
    # Write frames to video
    print("\n🎬 Constructing video...")
    for frame_file in tqdm(frame_files, desc="Writing frames"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            out.write(frame)
        else:
            print(f"\n⚠️  Warning: Could not read {frame_file}")
    
    out.release()
    
    # Verify output
    if os.path.exists(output_video_path):
        file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        duration = len(frame_files) / fps
        print(f"\n✅ Video created successfully!")
        print(f"📦 File size: {file_size_mb:.2f} MB")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"🎞️  Total frames: {len(frame_files)}")
    else:
        print(f"\n❌ Error: Video file was not created")
    
    print("=" * 70)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "output", "reconstructed_frames_v4")
    output_video_path = os.path.join(project_root, "output", "reconstructed_video_v4.mp4")
    
    construct_video(frames_dir, output_video_path, fps=30)


if __name__ == "__main__":
    main()
