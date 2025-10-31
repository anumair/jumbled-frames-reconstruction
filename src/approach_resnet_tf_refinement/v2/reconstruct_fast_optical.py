import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class FastOpticalFlowReconstructor:
    def __init__(self, frames_dir, output_dir):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, frame_path):
        """Extract ResNet-50 features from a frame"""
        img = Image.open(frame_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tensor)
        
        return features.cpu().numpy().flatten()
    
    def greedy_order_frames(self, features, start_idx=38):
        """Order frames using greedy nearest neighbor"""
        n_frames = len(features)
        
        visited = set()
        frame_order = [start_idx]
        visited.add(start_idx)
        
        current = start_idx
        
        print(f"\nStarting from frame {start_idx} (same as v1)")
        print("Ordering frames using greedy algorithm...")
        
        for _ in tqdm(range(n_frames - 1), desc="Ordering frames"):
            best_sim = -1
            best_next = None
            
            current_feat = features[current]
            
            # Find most similar unvisited frame
            for next_idx in range(n_frames):
                if next_idx in visited:
                    continue
                
                next_feat = features[next_idx]
                # Cosine similarity
                sim = np.dot(current_feat, next_feat) / (np.linalg.norm(current_feat) * np.linalg.norm(next_feat))
                
                if sim > best_sim:
                    best_sim = sim
                    best_next = next_idx
            
            frame_order.append(best_next)
            visited.add(best_next)
            current = best_next
        
        return frame_order
    
    def check_and_correct_direction(self, frame_paths, frame_order):
        """Use optical flow to check if person is walking forward or backward"""
        print("\n🔍 Checking walk direction with optical flow...")
        
        # Sample middle section of the video
        n_samples = 10
        start_idx = len(frame_order) // 3
        end_idx = 2 * len(frame_order) // 3
        sample_indices = np.linspace(start_idx, end_idx - 1, n_samples, dtype=int)
        
        horizontal_flows = []
        
        for i in sample_indices:
            idx1 = frame_order[i]
            idx2 = frame_order[i + 1]
            
            frame1 = cv2.imread(frame_paths[idx1])
            frame2 = cv2.imread(frame_paths[idx2])
            
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            horizontal_flow = np.mean(flow[..., 0])
            horizontal_flows.append(horizontal_flow)
        
        avg_horizontal_flow = np.mean(horizontal_flows)
        
        print(f"Average horizontal flow: {avg_horizontal_flow:.3f}")
        
        if avg_horizontal_flow < -1.0:  # Negative = walking backward
            print("⚠️  Person is walking backward! Reversing sequence...")
            frame_order = frame_order[::-1]
            print("✅ Sequence reversed - person now walking forward")
        else:
            print("✅ Direction correct - person walking forward")
        
        return frame_order
    
    def reconstruct(self):
        """Main reconstruction method"""
        print("=" * 60)
        print("FAST OPTICAL FLOW RECONSTRUCTION (V2)")
        print("=" * 60)
        
        # Get all frame paths
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.jpg')])
        frame_paths = [os.path.join(self.frames_dir, f) for f in frame_files]
        
        print(f"\n📊 Total frames: {len(frame_paths)}")
        
        # Extract ResNet features
        print("\n🔄 Extracting ResNet-50 features...")
        features = []
        for frame_path in tqdm(frame_paths, desc="Feature extraction"):
            feat = self.extract_features(frame_path)
            features.append(feat)
        features = np.array(features)
        
        # Order frames (using v1's starting point)
        frame_order = self.greedy_order_frames(features, start_idx=38)
        
        # Check direction and correct if needed
        frame_order = self.check_and_correct_direction(frame_paths, frame_order)
        
        # Copy frames in reconstructed order
        print("\n📁 Saving reconstructed frames...")
        for new_idx, original_idx in enumerate(tqdm(frame_order, desc="Copying frames")):
            src_path = frame_paths[original_idx]
            dst_path = os.path.join(self.output_dir, f"frame_{new_idx:04d}.jpg")
            img = cv2.imread(src_path)
            cv2.imwrite(dst_path, img)
        
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
