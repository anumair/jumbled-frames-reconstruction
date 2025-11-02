import os
import cv2
import numpy as np
from tqdm import tqdm

def reverse_v2_output():
    """
    Creates a reversed video from V2's output.
    Simply reverses the frame order from V2.
    """
    print("=" * 70)
    print("V3: REVERSE V2 OUTPUT")
    print("=" * 70)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Look for V2 output in the correct directory (v2_fixed_direction)
    v2_output_dir = os.path.join(project_root, "output", "reconstructed_frames_v2_fixed_direction")
    v2_order_file = os.path.join(v2_output_dir, "optical_refined_frames.txt")
    v3_output_dir = os.path.join(project_root, "output", "reconstructed_frames_v3_reversed")
    original_frames_dir = os.path.join(project_root, "frames")
    
    # Create output directory
    os.makedirs(v3_output_dir, exist_ok=True)
    
    # Load V2's frame order
    if not os.path.exists(v2_order_file):
        print(f"❌ V2 frame order not found: {v2_order_file}")
        return
    
    print(f"\n📂 Loading V2's frame order from: {v2_order_file}")
    v2_frame_order = np.loadtxt(v2_order_file, dtype=int).tolist()
    print(f"   Loaded {len(v2_frame_order)} frames")
    
    # Reverse the frame order
    reversed_order = v2_frame_order[::-1]
    print(f"\n🔄 Reversing frame order...")
    print(f"   Original first frame: {v2_frame_order[0]}")
    print(f"   Reversed first frame: {reversed_order[0]}")
    
    # Get original frame paths
    frame_files = sorted([f for f in os.listdir(original_frames_dir) 
                         if f.lower().endswith(('.jpg', '.png'))])
    frame_paths = [os.path.join(original_frames_dir, f) for f in frame_files]
    
    print(f"\n📁 Saving reversed frames to: {v3_output_dir}")
    
    # Copy frames in reversed order
    for new_idx, original_idx in enumerate(tqdm(reversed_order, desc="Copying frames")):
        if original_idx >= len(frame_paths):
            print(f"   ⚠️  Invalid index {original_idx}, skipping")
            continue
        
        src_path = frame_paths[original_idx]
        dst_path = os.path.join(v3_output_dir, f"frame_{new_idx:04d}.jpg")
        
        img = cv2.imread(src_path)
        if img is None:
            print(f"   ⚠️  Could not read {src_path}, skipping")
            continue
        
        cv2.imwrite(dst_path, img)
    
    # Save reversed frame order
    order_file = os.path.join(v3_output_dir, "frame_order_v3_reversed.txt")
    np.savetxt(order_file, np.array(reversed_order, dtype=int), fmt="%d")
    print(f"\n📝 Reversed frame order saved to: {order_file}")
    
    print("\n" + "=" * 70)
    print("✅ V3 REVERSE COMPLETE!")
    print("=" * 70)
    print(f"📂 Output: {v3_output_dir}")
    print(f"🎬 Video length: {len(reversed_order)} frames = {len(reversed_order)/30:.2f} seconds")
    
    return reversed_order


if __name__ == "__main__":
    reverse_v2_output()
