import cv2
import os
from tqdm import tqdm

def reconstruct_video_from_frames(frames_dir, output_video_path, fps=30):
    """
    Reconstruct video from ordered frames.
    
    Args:
        frames_dir: Directory containing ordered frames (frame_0000.jpg, frame_0001.jpg, etc.)
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
    """
    print("=" * 70)
    print("VIDEO RECONSTRUCTION (V5)")
    print("=" * 70)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.lower().endswith(('.jpg', '.png'))])
    
    if not frame_files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    
    print(f"\n📊 Found {len(frame_files)} frames")
    print(f"📹 Output FPS: {fps}")
    print(f"⏱️  Video duration: {len(frame_files)/fps:.2f} seconds")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {first_frame_path}")
    
    height, width, _ = first_frame.shape
    print(f"📐 Frame dimensions: {width}x{height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_video_path}")
    
    # Write frames to video
    print(f"\n🎬 Creating video...")
    for frame_file in tqdm(frame_files, desc="Writing frames"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"\n⚠️  Warning: Could not read {frame_file}, skipping...")
            continue
        
        out.write(frame)
    
    out.release()
    
    print(f"\n✅ Video saved to: {output_video_path}")
    print(f"📊 Total frames written: {len(frame_files)}")
    print("=" * 70)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "output", "reconstructed_frames_v5_1")
    output_video = os.path.join(project_root, "output", "reconstructed_v5.mp4")
    
    reconstruct_video_from_frames(frames_dir, output_video, fps=30)


if __name__ == "__main__":
    main()
