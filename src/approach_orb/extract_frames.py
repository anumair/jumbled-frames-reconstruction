import cv2
import os

def extract_frames(video_path, output_dir):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print("Error: Video file not found at", video_path)
        return

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    print("Extracting frames...")

    # Read and save frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Done. Extracted {frame_count} frames to: {output_dir}")


if __name__ == "__main__":
    # Full absolute path to your video
    video_path = r"input\jumbled_video.mp4"
    
    # Folder where frames will be saved
    output_dir = r"frames"

    extract_frames(video_path, output_dir)
