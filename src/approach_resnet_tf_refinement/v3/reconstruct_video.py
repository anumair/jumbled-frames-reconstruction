import cv2
import os
from tqdm import tqdm

def reconstruct_video(frames_dir, output_video_path, fps=30):
    """
    Reconstruct video from ordered frames
    """
    print("=" * 70)
    print("VIDEO RECONSTRUCTION (V3)")
    print("=" * 70)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.lower().endswith(('.jpg', '.png'))])
    
    if len(frame_files) == 0:
        print("❌ No frames found in directory")
        return
    
    print(f"\n📊 Total frames: {len(frame_files)}")
    print(f"🎬 Output video: {output_video_path}")
    print(f"⏱️  Frame rate: {fps} fps")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"❌ Could not read first frame: {first_frame_path}")
        return
    
    height, width = first_frame.shape[:2]
    print(f"📐 Resolution: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Could not open video writer")
        return
    
    # Write frames to video
    print("\n🎬 Writing frames to video...")
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            out.write(frame)
        else:
            print(f"⚠️  Warning: Could not read frame {frame_file}")
    
    out.release()
    
    print("\n" + "=" * 70)
    print("✅ VIDEO RECONSTRUCTION COMPLETE!")
    print("=" * 70)
    print(f"📂 Output: {output_video_path}")
    print(f"🎬 Total frames: {len(frame_files)}")
    print(f"⏱️  Duration: {len(frame_files)/fps:.2f} seconds")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "output", "reconstructed_frames_v3")
    output_video = os.path.join(project_root, "output", "reconstructed_v3.mp4")
    
    reconstruct_video(frames_dir, output_video, fps=30)


if __name__ == "__main__":
    main()
