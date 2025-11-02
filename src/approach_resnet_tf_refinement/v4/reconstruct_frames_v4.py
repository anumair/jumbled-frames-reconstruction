import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class HybridFrameReconstructor:
    def __init__(self, frames_dir, output_dir, save_order=True):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.save_order = save_order

        # Load ResNet-50
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
        """Extract ResNet-50 features"""
        img = Image.open(frame_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.cpu().numpy().flatten()

    def find_true_start_with_flow(self, frame_paths, features):
        """
        Find TRUE starting point by testing which direction produces most consistent flow.
        Tests multiple candidates and picks the one with best forward flow consistency.
        """
        print("\n🔍 Finding TRUE starting point using directional flow analysis...")
        
        n_frames = len(features)
        similarity_matrix = cosine_similarity(features)
        
        # Get top candidates based on multiple heuristics
        candidates = set()
        
        # Heuristic 1: Frames with low average similarity (likely endpoints)
        avg_sims = similarity_matrix.mean(axis=1)
        bottom_5_percent = int(n_frames * 0.05)  # Reduced from 10% to 5%
        low_sim_candidates = np.argsort(avg_sims)[:bottom_5_percent]
        candidates.update(low_sim_candidates.tolist())
        
        # Heuristic 2: Frames with asymmetric similarity distribution (sample only)
        sample_indices = np.linspace(0, n_frames - 1, min(50, n_frames), dtype=int)
        for i in sample_indices:
            sims_sorted = np.sort(similarity_matrix[i])[::-1]
            # Endpoint should have similarity skewed to one side
            skewness = sims_sorted[:20].mean() - sims_sorted[-20:].mean()
            if skewness > 0.15:  # High skewness = likely endpoint
                candidates.add(i)
        
        # Limit to top 30 candidates maximum
        if len(candidates) > 30:
            # Rank by average similarity (lower is better for endpoints)
            candidate_list = list(candidates)
            candidate_sims = [(c, avg_sims[c]) for c in candidate_list]
            candidate_sims.sort(key=lambda x: x[1])
            candidates = set([c for c, _ in candidate_sims[:30]])
        
        print(f"   Testing {len(candidates)} candidate start points...")
        
        best_start = None
        best_score = -float('inf')
        best_direction = None
        
        for candidate in tqdm(list(candidates), desc="   Evaluating candidates"):
            # Build short sequence from this candidate
            visited = set([candidate])
            current = candidate
            sequence = [candidate]
            
            # Greedy chain of 15 frames
            for _ in range(min(15, n_frames - 1)):
                sims = similarity_matrix[current].copy()
                sims[list(visited)] = -1
                next_idx = np.argmax(sims)
                if sims[next_idx] < 0.7:
                    break
                sequence.append(next_idx)
                visited.add(next_idx)
                current = next_idx
            
            if len(sequence) < 5:
                continue
            
            # Test FORWARD flow consistency from this sequence
            forward_flow = self.calculate_directional_flow(frame_paths, sequence)
            
            # Test BACKWARD flow consistency (reversed sequence)
            backward_flow = self.calculate_directional_flow(frame_paths, sequence[::-1])
            
            # Score: prefer consistent forward flow (positive) over backward (negative)
            # Also penalize high variance (inconsistent motion)
            forward_score = np.mean(forward_flow) + 2.0 / (np.std(forward_flow) + 0.1)
            backward_score = np.mean(backward_flow) + 2.0 / (np.std(backward_flow) + 0.1)
            
            # Pick the direction with better forward motion
            if forward_score > backward_score and forward_score > best_score:
                best_score = forward_score
                best_start = candidate
                best_direction = "forward"
            elif backward_score > forward_score and backward_score > best_score:
                best_score = backward_score
                best_start = sequence[-1]  # Start from end of sequence (reverse direction)
                best_direction = "backward (reversed)"
        
        if best_start is None:
            # Fallback to first frame
            best_start = 0
            best_direction = "fallback"
        
        print(f"   ✅ Selected start: Frame {best_start} (direction: {best_direction}, score: {best_score:.2f})")
        return best_start

    def calculate_directional_flow(self, frame_paths, sequence):
        """Calculate horizontal optical flow for a sequence"""
        flows = []
        for i in range(len(sequence) - 1):
            idx1 = sequence[i]
            idx2 = sequence[i + 1]
            
            if idx1 >= len(frame_paths) or idx2 >= len(frame_paths):
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
                flows.append(horizontal_flow)
            except:
                continue
        
        return flows if flows else [0]

    def greedy_order_frames(self, features, start_idx):
        """Simple greedy ordering (like V2) - smooth and reliable"""
        print("\n🔄 Ordering frames with greedy algorithm...")
        
        n_frames = len(features)
        similarity_matrix = cosine_similarity(features)
        
        visited = np.zeros(n_frames, dtype=bool)
        frame_order = [start_idx]
        visited[start_idx] = True
        current = start_idx
        
        norms = np.linalg.norm(features, axis=1)
        norms[norms == 0] = 1e-6
        
        for _ in tqdm(range(n_frames - 1), desc="   Building sequence"):
            current_feat = features[current]
            sims = features.dot(current_feat) / (norms * np.linalg.norm(current_feat) + 1e-12)
            sims[visited] = -np.inf
            best_next = int(np.argmax(sims))
            
            if sims[best_next] == -np.inf:
                break
            
            frame_order.append(best_next)
            visited[best_next] = True
            current = best_next
        
        print(f"   ✅ Ordered {len(frame_order)} frames")
        return frame_order

    def detect_sequence_break(self, frame_paths, frame_order, similarity_matrix):
        """
        Detect where the GOOD sequence ends and the BAD tail begins.
        Uses similarity-based analysis only (faster than flow).
        """
        print("\n🔍 Detecting sequence break point...")
        
        n = len(frame_order)
        if n < 20:
            return n  # Keep all frames if too short
        
        # Calculate similarity scores for consecutive frames
        similarity_scores = []
        for i in range(n - 1):
            sim = similarity_matrix[frame_order[i], frame_order[i + 1]]
            similarity_scores.append(sim)
        
        similarity_scores = np.array(similarity_scores)
        
        # Find where similarity drops significantly
        # Use derivative to find sharp drops
        quality_derivative = np.diff(similarity_scores)
        
        # Look for the sharpest drop in the last 30% of video
        search_start = int(len(quality_derivative) * 0.7)
        if search_start < len(quality_derivative):
            drop_candidates = quality_derivative[search_start:]
            if len(drop_candidates) > 0:
                sharpest_drop_idx = search_start + np.argmin(drop_candidates)
                
                # Verify this is actually a significant drop
                if quality_derivative[sharpest_drop_idx] < -0.15:
                    cutoff = sharpest_drop_idx + 2
                    print(f"   🎯 Break detected at frame {cutoff}/{n}")
                    print(f"   Quality drop: {quality_derivative[sharpest_drop_idx]:.3f}")
                    return cutoff
        
        # Fallback: find where similarity first drops below threshold in last 20%
        threshold = np.percentile(similarity_scores, 25)  # Bottom 25%
        search_start = int(len(similarity_scores) * 0.75)
        for i in range(search_start, len(similarity_scores)):
            if similarity_scores[i] < threshold:
                cutoff = i + 1
                print(f"   🎯 Break detected at frame {cutoff}/{n} (similarity threshold: {threshold:.3f})")
                return cutoff
        
        # No clear break found - keep all
        print(f"   ℹ️  No clear break detected - keeping all frames")
        return n

    def verify_and_reverse_if_needed(self, frame_paths, frame_order):
        """Check if sequence should be reversed (like V2)"""
        print("\n🔄 Verifying direction with optical flow...")
        
        n = len(frame_order)
        if n < 2:
            return frame_order
        
        # Sample from middle 50% of sequence
        start_idx = n // 4
        end_idx = 3 * n // 4
        sample_size = min(20, (end_idx - start_idx) // 2)
        sample_indices = np.linspace(start_idx, end_idx - 1, sample_size, dtype=int)
        
        flows = []
        for i in sample_indices:
            idx1 = frame_order[i]
            idx2 = frame_order[i + 1]
            
            if idx1 >= len(frame_paths) or idx2 >= len(frame_paths):
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
                flows.append(horizontal_flow)
            except:
                continue
        
        if not flows:
            return frame_order
        
        median_flow = np.median(flows)
        mad = np.median(np.abs(np.array(flows) - median_flow))
        
        print(f"   Median flow: {median_flow:.4f} (MAD: {mad:.4f})")
        
        threshold = max(0.5, 1.5 * mad)
        if median_flow < -threshold:
            print(f"   🔄 Reversing sequence (backward motion)")
            return frame_order[::-1]
        
        print(f"   ✅ Direction confirmed")
        return frame_order

    def reconstruct(self):
        """Main reconstruction pipeline"""
        print("=" * 70)
        print("HYBRID FRAME RECONSTRUCTION (V4)")
        print("V2 Smoothness + Intelligent Tail Removal")
        print("=" * 70)
        
        # Load frames
        frame_files = sorted([f for f in os.listdir(self.frames_dir) 
                            if f.lower().endswith(('.jpg', '.png'))])
        frame_paths = [os.path.join(self.frames_dir, f) for f in frame_files]
        print(f"\n📊 Total frames: {len(frame_paths)}")
        
        if len(frame_paths) == 0:
            raise RuntimeError("No frames found")
        
        # Extract features
        print("\n🔄 Extracting ResNet-50 features...")
        features = []
        for frame_path in tqdm(frame_paths, desc="Feature extraction"):
            feat = self.extract_features(frame_path)
            features.append(feat)
        features = np.array(features)
        
        # Find optimal start with flow analysis
        start_idx = self.find_true_start_with_flow(frame_paths, features)
        
        # Order frames (V2 style - smooth)
        frame_order = self.greedy_order_frames(features, start_idx)
        
        # Verify direction
        frame_order = self.verify_and_reverse_if_needed(frame_paths, frame_order)
        
        # Detect and remove bad tail
        similarity_matrix = cosine_similarity(features)
        cutoff_point = self.detect_sequence_break(frame_paths, frame_order, similarity_matrix)
        
        original_length = len(frame_order)
        frame_order = frame_order[:cutoff_point]
        
        print(f"\n📊 Sequence trimmed: {original_length} → {len(frame_order)} frames")
        print(f"   Removed {original_length - len(frame_order)} problematic frames from tail")
        
        # Save reconstructed frames
        print("\n📁 Saving reconstructed frames...")
        for new_idx, original_idx in enumerate(tqdm(frame_order, desc="Copying frames")):
            if original_idx >= len(frame_paths):
                continue
            src_path = frame_paths[original_idx]
            dst_path = os.path.join(self.output_dir, f"frame_{new_idx:04d}.jpg")
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)
        
        # Save order
        if self.save_order:
            order_path = os.path.join(self.output_dir, "frame_order_v4.txt")
            np.savetxt(order_path, np.array(frame_order, dtype=int), fmt="%d")
            print(f"📝 Frame order saved to: {order_path}")
        
        print("\n" + "=" * 70)
        print("✅ RECONSTRUCTION COMPLETE!")
        print("=" * 70)
        print(f"📂 Output: {self.output_dir}")
        print(f"🎬 Final video: {len(frame_order)} frames")
        
        return frame_order


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "frames")
    output_dir = os.path.join(project_root, "output", "reconstructed_frames_v4")
    
    reconstructor = HybridFrameReconstructor(frames_dir, output_dir)
    reconstructor.reconstruct()


if __name__ == "__main__":
    main()
