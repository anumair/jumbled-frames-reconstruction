import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

class FastOpticalFlowReconstructor:
    def __init__(self, frames_dir, output_dir, start_idx=38, save_order=True):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Config
        self.start_idx = start_idx
        self.save_order = save_order

        # Load ResNet-50 for feature extraction
        print("Loading ResNet-50 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, frame_path):
        """Extract ResNet-50 features from a frame"""
        img = Image.open(frame_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.cpu().numpy().flatten()

    def greedy_order_frames(self, features, frame_paths, start_idx=None):
        """Order frames using greedy nearest neighbor (safe against bad start_idx)"""
        if start_idx is None:
            start_idx = self.start_idx

        n_frames = len(features)
        if n_frames == 0:
            return []

        # clamp start_idx
        start_idx = int(min(max(0, start_idx), n_frames - 1))

        visited = np.zeros(n_frames, dtype=bool)
        frame_order = [start_idx]
        visited[start_idx] = True
        current = start_idx

        print(f"\nStarting from frame {start_idx} (same as v1)")
        print("Ordering frames using greedy algorithm...")

        # Precompute norms to avoid division by zero
        norms = np.linalg.norm(features, axis=1)
        norms[norms == 0] = 1e-6

        for _ in tqdm(range(n_frames - 1), desc="Ordering frames"):
            current_feat = features[current]
            # compute cosine similarity vectorized
            sims = features.dot(current_feat) / (norms * np.linalg.norm(current_feat) + 1e-12)
            sims[visited] = -np.inf  # exclude visited
            best_next = int(np.argmax(sims))
            if sims[best_next] == -np.inf:
                # no candidate left (shouldn't happen) -> break
                break
            frame_order.append(best_next)
            visited[best_next] = True
            current = best_next

        return frame_order

    def refine_order_with_optical_flow(self, frame_paths, frame_order):
        """Refine the order using optical flow to ensure forward motion"""
        print("\n🔄 Refining order with optical flow...")

        n = len(frame_order)
        if n < 2:
            print("Not enough frames to refine with optical flow.")
            return frame_order

        # adaptive check interval (scale with n)
        check_interval = max(1, n // 30)  # ~30 samples across video
        sample_indices = list(range(0, n - 1, check_interval))
        horizontal_flow_samples = []

        for i in sample_indices:
            idx1 = frame_order[i]
            idx2 = frame_order[i + 1]

            # defensive frame read
            if idx1 < 0 or idx2 < 0 or idx1 >= len(frame_paths) or idx2 >= len(frame_paths):
                continue
            frame1 = cv2.imread(frame_paths[idx1])
            frame2 = cv2.imread(frame_paths[idx2])
            if frame1 is None or frame2 is None:
                continue

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # compute Farneback flow
            try:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                horizontal_flow = float(np.median(flow[..., 0]))  # median is more robust to noise
                horizontal_flow_samples.append(horizontal_flow)
            except Exception as e:
                # if Farneback fails for this pair, skip
                continue

        if len(horizontal_flow_samples) == 0:
            print("No valid optical flow samples found — skipping optical-flow based reversal.")
            return frame_order

        # Use median sign to decide direction (robust)
        med_flow = np.median(horizontal_flow_samples)
        mad = np.median(np.abs(horizontal_flow_samples - med_flow))  # robust spread
        print(f"Optical-flow samples: {len(horizontal_flow_samples)}, median={med_flow:.4f}, MAD={mad:.4f}")

        # adaptive threshold: require median to be meaningfully negative to consider backward
        threshold = max(0.5, 1.5 * mad)  # min threshold 0.5 px, scaled with MAD
        if med_flow < -threshold:
            print(f"⚠️ Detected backward motion (median < -{threshold:.3f}) -> Reversing frame order")
            frame_order = frame_order[::-1]
            print("✅ Frame order reversed")
        else:
            print("✅ Forward motion detected - order looks good")

        return frame_order

    def verify_direction(self, frame_paths, frame_order):
        """Verify the walk direction is correct using more samples"""
        print("\n🔍 Verifying walk direction...")

        n = len(frame_order)
        if n < 2:
            print("Not enough frames to verify direction.")
            return frame_order

        # sample up to 20 pairs from middle half
        start_idx = n // 4
        end_idx = 3 * n // 4
        num_samples = min(20, max(2, (end_idx - start_idx) // 2))
        sample_indices = np.linspace(start_idx, end_idx - 1, num_samples, dtype=int)

        horizontal_flows = []

        for i in sample_indices:
            idx1 = frame_order[i]
            idx2 = frame_order[i + 1]
            if idx1 < 0 or idx2 < 0 or idx1 >= len(frame_paths) or idx2 >= len(frame_paths):
                continue

            frame1 = cv2.imread(frame_paths[idx1])
            frame2 = cv2.imread(frame_paths[idx2])
            if frame1 is None or frame2 is None:
                continue

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            try:
                flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                horizontal_flow = float(np.median(flow[..., 0]))
                horizontal_flows.append(horizontal_flow)
            except Exception:
                continue

        if len(horizontal_flows) == 0:
            print("No valid optical flow samples found for verification.")
            return frame_order

        avg_horizontal_flow = float(np.mean(horizontal_flows))
        print(f"Average horizontal flow (verification samples): {avg_horizontal_flow:.4f}")

        if avg_horizontal_flow > 0:
            print("✅ Direction verified - person walking forward")
        else:
            print("⚠️ Warning: Person may still be walking backward")

        return frame_order

    def reconstruct(self, save_order_path=None):
        """Main reconstruction method"""
        print("=" * 60)
        print("FAST OPTICAL FLOW RECONSTRUCTION (V2)")
        print("=" * 60)

        # Get all frame paths
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.lower().endswith('.jpg')])
        frame_paths = [os.path.join(self.frames_dir, f) for f in frame_files]
        print(f"\n📊 Total frames: {len(frame_paths)}")

        if len(frame_paths) == 0:
            raise RuntimeError("No frames found in frames_dir")

        # Extract ResNet features
        print("\n🔄 Extracting ResNet-50 features...")
        features = []
        for frame_path in tqdm(frame_paths, desc="Feature extraction"):
            feat = self.extract_features(frame_path)
            features.append(feat)
        features = np.array(features)

        # Order frames (using v1's starting point)
        frame_order = self.greedy_order_frames(features, frame_paths, start_idx=self.start_idx)

        # Refine with optical flow check
        frame_order = self.refine_order_with_optical_flow(frame_paths, frame_order)

        # Final verification
        frame_order = self.verify_direction(frame_paths, frame_order)

        # Copy frames in reconstructed order
        print("\n📁 Saving reconstructed frames...")
        for new_idx, original_idx in enumerate(tqdm(frame_order, desc="Copying frames")):
            if original_idx < 0 or original_idx >= len(frame_paths):
                print(f"⚠️ invalid index {original_idx} in final order, skipping")
                continue
            src_path = frame_paths[original_idx]
            dst_path = os.path.join(self.output_dir, f"frame_{new_idx:04d}.jpg")
            img = cv2.imread(src_path)
            if img is None:
                print(f"⚠️ Could not read {src_path}, skipping")
                continue
            cv2.imwrite(dst_path, img)

        # Optionally save final ordering to a file (path or default)
        if save_order_path is None:
            save_order_path = os.path.join(self.output_dir, "optical_refined_frames.txt")

        if self.save_order:
            np.savetxt(save_order_path, np.array(frame_order, dtype=int), fmt="%d")
            print(f"📝 Final frame order saved to: {save_order_path}")

        print(f"\n✅ Reconstruction complete!")
        print(f"📂 Output: {self.output_dir}")
        return frame_order


def main():
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    frames_dir = os.path.join(project_root, "frames")
    output_dir = os.path.join(project_root, "output", "reconstructed_frames_optical_flow_v2")

    reconstructor = FastOpticalFlowReconstructor(frames_dir, output_dir)
    reconstructor.reconstruct()


if __name__ == "__main__":
    main()
