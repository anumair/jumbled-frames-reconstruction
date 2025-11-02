import cv2
import os
from tqdm import tqdm

def reconstruct_video(frames_dir, frame_order_file, output_video, fps=30):
    """
    Reconstruct video from ordered frames.
    
    Args:
        frames_dir: Directory containing frame images
        frame_order_file: Path to file containing frame order
        output_video: Path to output video file
        fps: Frames per second for output video
    """
    # Load frame order
    print("Loading frame order...")
    with open(frame_order_file, 'r') as f:
        frame_order = [int(line.strip()) for line in f.readlines()]
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Reconstructing video: {len(frame_order)} frames at {fps} FPS")
    print(f"Output: {output_video}")
    
    # Write frames in order
    for idx in tqdm(frame_order, desc="Reconstructing video"):
        frame_path = os.path.join(frames_dir, frame_files[idx])
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"✅ Video saved to '{output_video}'")

if __name__ == "__main__":
    frames_dir = "../../../frames"
    frame_order_file = "frame_order_refined.txt"
    output_video = "../../../output/reconstructed_resnet_tf_refined_v1.mp4"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs("../../../output", exist_ok=True)
    
    reconstruct_video(frames_dir, frame_order_file, output_video)
