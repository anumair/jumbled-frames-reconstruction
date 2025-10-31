import cv2
import os
from tqdm import tqdm

def create_video_from_frames(frames_dir, output_video, fps=30):
    """Create video from reconstructed frames"""
    print("\n" + "="*60)
    print("CREATING VIDEO FROM OPTICAL FLOW RECONSTRUCTION")
    print("="*60)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print("❌ No frames found!")
        return
    
    print(f"\n📊 Total frames: {len(frame_files)}")
    print(f"🎬 FPS: {fps}")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    print(f"📐 Resolution: {width}x{height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write frames to video
    print(f"\n🎥 Writing frames to video...")
    for frame_file in tqdm(frame_files, desc="Creating video"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    
    print(f"\n✅ Video created successfully!")
    print(f"📂 Output: {output_video}")
    
    # Get file size
    file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
    print(f"📦 Size: {file_size:.2f} MB")


def main():
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "output", "reconstructed_frames_optical_flow_v2")
    output_video = os.path.join(project_root, "output", "reconstructed_optical_flow_v2.mp4")
    
    create_video_from_frames(frames_dir, output_video, fps=30)


if __name__ == "__main__":
    main()
