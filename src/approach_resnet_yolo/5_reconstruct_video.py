import cv2
import os
from tqdm import tqdm

def reconstruct_video(frames_dir, frame_order_file, output_video, fps=30):
    """
    Reconstruct video from ordered frames.
    
    Args:
        frames_dir: Directory containing extracted frames
        frame_order_file: Path to frame order file
        output_video: Output video path
        fps: Frames per second
    """
    # Read frame order
    with open(frame_order_file, 'r') as f:
        frame_order = [int(line.strip()) for line in f]
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[frame_order[0]])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write frames in order
    for idx in tqdm(frame_order, desc="Reconstructing video"):
        frame_path = os.path.join(frames_dir, frame_files[idx])
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"✅ Video reconstructed and saved to '{output_video}'")

if __name__ == "__main__":
    frames_dir = "frames"
    frame_order_file = "src/approach_resnet_yolo/frame_order.txt"
    output_video = "output/reconstructed_resnet_yolo.mp4"
    
    os.makedirs("output", exist_ok=True)
    reconstruct_video(frames_dir, frame_order_file, output_video)
