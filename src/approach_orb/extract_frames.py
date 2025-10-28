import cv2
import os

def extract_frames(video_path, output_dir):
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_count += 1

    cap.release()
    print(f"✅ Extracted {frame_count} frames to '{output_dir}'.")

if __name__ == "__main__":
    video_path = "jumbled_video.mp4"
    output_dir = "frames"
    extract_frames(video_path, output_dir)
