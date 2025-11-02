"""
Complete Pipeline for Frame Reconstruction
Runs V1 -> V2 -> V3 sequentially to produce final unjumbled video
"""

import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """Run a Python script and measure execution time"""
    print("\n" + "=" * 70)
    print(f"🔄 {description}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Get the virtual environment Python executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
        
        # Use venv python if it exists, otherwise use system python
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Run the script using subprocess
        result = subprocess.run(
            [python_exe, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed_time:.2f} seconds")
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        return False, elapsed_time

def main():
    """Main pipeline execution"""
    print("=" * 70)
    print("COMPLETE RECONSTRUCTION PIPELINE")
    print("V1 (ResNet50 + Refinement) → V2 (Optical Flow) → V3 (Final Video)")
    print("=" * 70)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Track total time and results
    total_start = time.time()
    results = []
    
    # ========== V0: Frame Extraction ==========
    print("\n\n📁 STAGE 1: V0 - Frame Extraction from Video")
    print("-" * 70)

    v1_dir = os.path.join(script_dir, "v1")
    success, elapsed = run_script(os.path.join(v1_dir, "0_extract_frames.py"), "V0: Extracting video frames")
    results.append(("V0: Extracting video frames", success, elapsed))

    if not success:
        print(f"\n❌ Pipeline failed at: V0 Frame Extraction")
        return

    # ========== V1: ResNet50 + Refinement ==========
    print("\n\n📁 STAGE 2: V1 - ResNet50 Feature Extraction & Refinement")
    print("-" * 70)
    
    v1_scripts = [
        ("1_extract_resnet_features.py", "V1: Extract ResNet50 features"),
        ("2_compute_similarity.py", "V1: Compute similarity matrix"),
        ("3_order_frames_smart.py", "V1: Smart frame ordering"),
        ("4_refine_ordering.py", "V1: Refine ordering with 2-opt + sliding window"),
    ]
    
    for script_name, description in v1_scripts:
        script_path = os.path.join(v1_dir, script_name)
        success, elapsed = run_script(script_path, description)
        results.append((description, success, elapsed))
        
        if not success:
            print(f"\n❌ Pipeline failed at: {description}")
            return
    
    # ========== V2: Optical Flow ==========
    print("\n\n📁 STAGE 3: V2 - Optical Flow Refinement")
    print("-" * 70)
    
    v2_dir = os.path.join(script_dir, "v2")
    v2_script = os.path.join(v2_dir, "reconstruct_fast_optical.py")
    success, elapsed = run_script(v2_script, "V2: Optical flow refinement")
    results.append(("V2: Optical flow refinement", success, elapsed))
    
    if not success:
        print(f"\n❌ Pipeline failed at: V2 Optical Flow")
        return
    
    # ========== V3: Final Video ==========
    print("\n\n📁 STAGE 4: V3 - Final Video Reconstruction")
    print("-" * 70)
    
    v3_dir = os.path.join(script_dir, "v3")
    v3_scripts = [
        ("reverse_video.py", "V3: Reverse frame order"),
        ("reconstruct_video.py", "V3: Create final unjumbled video"),
    ]
    
    for script_name, description in v3_scripts:
        script_path = os.path.join(v3_dir, script_name)
        success, elapsed = run_script(script_path, description)
        results.append((description, success, elapsed))
        
        if not success:
            print(f"\n❌ Pipeline failed at: {description}")
            return
    
    # ========== Summary ==========
    total_elapsed = time.time() - total_start
    
    print("\n\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n📊 Execution Summary:")
    print("-" * 70)
    for description, success, elapsed in results:
        status = "✅" if success else "❌"
        print(f"{status} {description}: {elapsed:.2f}s")
    
    print("-" * 70)
    print(f"⏱️  Total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    
    # Show output location
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_video = os.path.join(project_root, "output", "unjumbled_video.mp4")
    print(f"\n🎬 Final output video: {output_video}")
    
    if os.path.exists(output_video):
        print("✅ Video file exists and ready to view!")
    else:
        print("⚠️  Video file not found - check logs above for errors")

if __name__ == "__main__":
    main()
