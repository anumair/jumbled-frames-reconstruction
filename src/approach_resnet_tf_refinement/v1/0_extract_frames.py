
import os
import cv2
import shutil

def extract_frames(video_path, output_dir, image_format='jpg'):
    """
    Extracts all frames from a video and saves them to a directory.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        image_format (str): Format to save the frames (e.g., 'jpg', 'png').
    """
    print(f"Extracting frames from {video_path} to {output_dir}...")

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    # Clean up the output directory if it exists, then create it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"   Removed existing directory: {output_dir}")
    os.makedirs(output_dir)
    print(f"   Created directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"frame_{frame_count:04d}.{image_format}"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"✅ Successfully extracted {frame_count} frames.")

def main():
    """
    Main function to define paths and run the extraction.
    """
    # Assuming this script is in src/approach_resnet_tf_refinement/v1
    # Project root is three levels up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    # Define input video path
    input_video = os.path.join(project_root, 'input', 'jumbled_video.mp4')
    
    # Define output directory for frames
    frames_dir = os.path.join(project_root, 'frames')

    # Run the extraction
    extract_frames(input_video, frames_dir)

if __name__ == "__main__":
    print("======================================================================")
    print("V0: Extract Frames from Video")
    print("======================================================================")
    main()
