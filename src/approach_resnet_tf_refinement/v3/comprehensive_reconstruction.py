import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components

class ComprehensiveFrameReconstructor:
    def __init__(self, frames_dir, output_dir, save_order=True):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
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

    def detect_outliers(self, features, threshold_percentile=10):
        """
        Detect outlier frames based on average similarity to all other frames.
        Frames with low average similarity are likely outliers (blurry, transitions, etc.)
        """
        print("\n🔍 Step 1: Detecting outlier frames...")
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(features)
        
        # Calculate average similarity for each frame (excluding self-similarity)
        np.fill_diagonal(similarity_matrix, 0)
        avg_similarities = similarity_matrix.mean(axis=1)
        
        # Determine threshold using percentile
        threshold = np.percentile(avg_similarities, threshold_percentile)
        
        # Identify outliers
        outliers = avg_similarities < threshold
        outlier_indices = np.where(outliers)[0]
        inlier_indices = np.where(~outliers)[0]
        
        print(f"   Average similarity range: [{avg_similarities.min():.4f}, {avg_similarities.max():.4f}]")
        print(f"   Outlier threshold (P{threshold_percentile}): {threshold:.4f}")
        print(f"   Found {len(outlier_indices)} outliers ({len(outlier_indices)/len(features)*100:.1f}%)")
        print(f"   Keeping {len(inlier_indices)} inlier frames for reconstruction")
        
        return inlier_indices, outlier_indices, similarity_matrix

    def find_largest_connected_component(self, similarity_matrix, inlier_indices, similarity_threshold=0.75):
        """
        Find the largest connected component of frames with high similarity.
        This ensures we only reconstruct the main continuous sequence.
        """
        print("\n🔍 Step 2: Finding largest connected component...")
        
        # Create adjacency matrix for inliers only
        n_inliers = len(inlier_indices)
        adj_matrix = np.zeros((n_inliers, n_inliers), dtype=bool)
        
        for i in range(n_inliers):
            for j in range(n_inliers):
                if i != j:
                    sim = similarity_matrix[inlier_indices[i], inlier_indices[j]]
                    adj_matrix[i, j] = sim >= similarity_threshold
        
        # Find connected components
        n_components, labels = connected_components(adj_matrix, directed=False)
        
        print(f"   Similarity threshold for connectivity: {similarity_threshold}")
        print(f"   Found {n_components} connected components")
        
        # Find largest component
        component_sizes = np.bincount(labels)
        largest_component = np.argmax(component_sizes)
        largest_size = component_sizes[largest_component]
        
        print(f"   Largest component size: {largest_size} frames ({largest_size/n_inliers*100:.1f}% of inliers)")
        
        # Get indices of frames in largest component
        component_mask = labels == largest_component
        main_sequence_indices = inlier_indices[component_mask]
        
        return main_sequence_indices

    def find_optimal_start_with_optical_flow(self, frame_paths, valid_indices, similarity_matrix):
        """
        Find optimal starting point using both similarity and optical flow.
        True start should have minimal incoming optical flow.
        """
        print("\n🔍 Step 3: Finding optimal starting point with optical flow...")
        
        n_valid = len(valid_indices)
        if n_valid < 2:
            return valid_indices[0] if n_valid > 0 else 0
        
        # Sample candidates (test every 10th frame for speed)
        sample_step = max(1, n_valid // 20)
        candidate_indices = valid_indices[::sample_step]
        
        best_start = valid_indices[0]
        best_score = -float('inf')
        
        print(f"   Testing {len(candidate_indices)} candidate start points...")
        
        for candidate_idx in tqdm(candidate_indices, desc="   Evaluating starts"):
            if candidate_idx >= len(frame_paths):
                continue
                
            # Build short greedy chain from this candidate
            visited = set([candidate_idx])
            current = candidate_idx
            chain_similarity = 0
            chain_frames = [candidate_idx]
            
            # Build chain of 10 frames
            for _ in range(min(10, n_valid - 1)):
                # Get similarities to unvisited valid frames
                sims = []
                candidates = []
                for idx in valid_indices:
                    if idx not in visited:
                        sim = similarity_matrix[current, idx]
                        sims.append(sim)
                        candidates.append(idx)
                
                if not sims:
                    break
                
                next_idx = candidates[np.argmax(sims)]
                max_sim = max(sims)
                
                if max_sim < 0.7:  # Threshold
                    break
                
                chain_similarity += max_sim
                visited.add(next_idx)
                chain_frames.append(next_idx)
                current = next_idx
            
            # Calculate optical flow consistency for this chain
            if len(chain_frames) >= 3:
                flow_consistency = self.calculate_chain_flow_consistency(frame_paths, chain_frames)
                
                # Combined score: similarity + flow consistency
                score = chain_similarity + flow_consistency * 5  # Weight flow consistency
                
                if score > best_score:
                    best_score = score
                    best_start = candidate_idx
        
        print(f"   ✅ Selected starting frame: {best_start} (score: {best_score:.2f})")
        return best_start

    def calculate_chain_flow_consistency(self, frame_paths, chain_indices):
        """Calculate how consistent the optical flow is across a chain"""
        if len(chain_indices) < 2:
            return 0
        
        flows = []
        for i in range(len(chain_indices) - 1):
            idx1 = chain_indices[i]
            idx2 = chain_indices[i + 1]
            
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
            return 0
        
        # Consistency = negative of standard deviation (more consistent = higher score)
        consistency = -np.std(flows)
        return consistency

    def greedy_order_with_threshold(self, features, valid_indices, similarity_matrix, 
                                    start_idx, min_similarity=0.65):
        """
        Order frames using greedy nearest neighbor with similarity threshold.
        Stops adding frames when similarity drops below threshold.
        """
        print("\n🔄 Step 4: Ordering frames with similarity threshold...")
        
        n_valid = len(valid_indices)
        visited = set([start_idx])
        frame_order = [start_idx]
        current = start_idx
        
        # Create mapping from global index to valid index
        valid_set = set(valid_indices)
        
        low_similarity_count = 0
        max_low_count = 3  # Allow 3 consecutive low similarities before stopping
        
        pbar = tqdm(total=n_valid-1, desc="   Ordering frames")
        
        while len(frame_order) < n_valid:
            # Get similarities to unvisited valid frames
            best_sim = -1
            best_next = None
            
            for idx in valid_indices:
                if idx not in visited:
                    sim = similarity_matrix[current, idx]
                    if sim > best_sim:
                        best_sim = sim
                        best_next = idx
            
            # Check if we should continue
            if best_next is None:
                print(f"\n   ⚠️  No more valid frames to add")
                break
            
            if best_sim < min_similarity:
                low_similarity_count += 1
                if low_similarity_count >= max_low_count:
                    print(f"\n   🛑 Stopping: Similarity dropped below {min_similarity:.3f} for {max_low_count} consecutive frames")
                    print(f"   Excluded {n_valid - len(frame_order)} low-similarity frames")
                    break
            else:
                low_similarity_count = 0  # Reset counter
            
            frame_order.append(best_next)
            visited.add(best_next)
            current = best_next
            pbar.update(1)
        
        pbar.close()
        
        print(f"   ✅ Ordered {len(frame_order)}/{n_valid} frames ({len(frame_order)/n_valid*100:.1f}%)")
        return frame_order

    def verify_and_refine_with_optical_flow(self, frame_paths, frame_order):
        """Verify direction and remove temporal inconsistencies"""
        print("\n🔄 Step 5: Verifying with optical flow...")
        
        n = len(frame_order)
        if n < 2:
            return frame_order
        
        # Sample pairs from entire sequence
        sample_size = min(30, n // 2)
        sample_indices = np.linspace(0, n - 2, sample_size, dtype=int)
        
        horizontal_flows = []
        
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
                horizontal_flows.append(horizontal_flow)
            except:
                continue
        
        if not horizontal_flows:
            print("   ⚠️  No valid optical flow samples")
            return frame_order
        
        median_flow = np.median(horizontal_flows)
        mad = np.median(np.abs(np.array(horizontal_flows) - median_flow))
        
        print(f"   Median horizontal flow: {median_flow:.4f} (MAD: {mad:.4f})")
        
        # Reverse if moving backward
        threshold = max(0.5, 1.5 * mad)
        if median_flow < -threshold:
            print(f"   🔄 Reversing sequence (backward motion detected)")
            frame_order = frame_order[::-1]
        else:
            print(f"   ✅ Direction confirmed (forward motion)")
        
        return frame_order

    def remove_temporal_outliers(self, frame_paths, frame_order, similarity_matrix, threshold=0.6):
        """
        Remove frames that have abnormally low similarity to their neighbors.
        This catches remaining outliers that slipped through.
        """
        print("\n🔄 Step 6: Removing temporal outliers...")
        
        if len(frame_order) < 3:
            return frame_order
        
        cleaned_order = [frame_order[0]]
        removed_count = 0
        
        for i in range(1, len(frame_order) - 1):
            prev_idx = frame_order[i - 1]
            curr_idx = frame_order[i]
            next_idx = frame_order[i + 1]
            
            # Check similarity to neighbors
            sim_to_prev = similarity_matrix[curr_idx, prev_idx]
            sim_to_next = similarity_matrix[curr_idx, next_idx]
            avg_neighbor_sim = (sim_to_prev + sim_to_next) / 2
            
            # Keep frame if it's similar enough to neighbors
            if avg_neighbor_sim >= threshold:
                cleaned_order.append(curr_idx)
            else:
                removed_count += 1
        
        # Always keep last frame
        cleaned_order.append(frame_order[-1])
        
        print(f"   Removed {removed_count} temporal outliers")
        print(f"   Final sequence length: {len(cleaned_order)} frames")
        
        return cleaned_order

    def reconstruct(self):
        """Main reconstruction pipeline"""
        print("=" * 70)
        print("COMPREHENSIVE FRAME RECONSTRUCTION (V3)")
        print("=" * 70)
        
        # Get all frame paths
        frame_files = sorted([f for f in os.listdir(self.frames_dir) 
                            if f.lower().endswith(('.jpg', '.png'))])
        frame_paths = [os.path.join(self.frames_dir, f) for f in frame_files]
        print(f"\n📊 Total frames found: {len(frame_paths)}")
        
        if len(frame_paths) == 0:
            raise RuntimeError("No frames found in frames_dir")
        
        # Extract features
        print("\n🔄 Extracting ResNet-50 features...")
        features = []
        for frame_path in tqdm(frame_paths, desc="Feature extraction"):
            feat = self.extract_features(frame_path)
            features.append(feat)
        features = np.array(features)
        
        # Step 1: Detect and remove outliers
        inlier_indices, outlier_indices, similarity_matrix = self.detect_outliers(features)
        
        # Step 2: Find largest connected component
        main_sequence_indices = self.find_largest_connected_component(
            similarity_matrix, inlier_indices, similarity_threshold=0.72
        )
        
        # Step 3: Find optimal start point
        start_idx = self.find_optimal_start_with_optical_flow(
            frame_paths, main_sequence_indices, similarity_matrix
        )
        
        # Step 4: Order frames with threshold
        frame_order = self.greedy_order_with_threshold(
            features, main_sequence_indices, similarity_matrix, 
            start_idx, min_similarity=0.65
        )
        
        # Step 5: Verify and refine with optical flow
        frame_order = self.verify_and_refine_with_optical_flow(frame_paths, frame_order)
        
        # Step 6: Remove temporal outliers
        frame_order = self.remove_temporal_outliers(
            frame_paths, frame_order, similarity_matrix, threshold=0.6
        )
        
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
        
        # Save frame order
        if self.save_order:
            order_path = os.path.join(self.output_dir, "frame_order_v3.txt")
            np.savetxt(order_path, np.array(frame_order, dtype=int), fmt="%d")
            print(f"📝 Frame order saved to: {order_path}")
        
        # Save outlier report
        report_path = os.path.join(self.output_dir, "reconstruction_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== RECONSTRUCTION REPORT (V3) ===\n\n")
            f.write(f"Total frames: {len(frame_paths)}\n")
            f.write(f"Outliers detected: {len(outlier_indices)}\n")
            f.write(f"Main sequence frames: {len(main_sequence_indices)}\n")
            f.write(f"Final reconstructed frames: {len(frame_order)}\n")
            f.write(f"Reconstruction rate: {len(frame_order)/len(frame_paths)*100:.1f}%\n\n")
            f.write(f"Outlier frame indices: {sorted(outlier_indices.tolist())}\n")
        print(f"📊 Report saved to: {report_path}")
        
        print("\n" + "=" * 70)
        print("✅ RECONSTRUCTION COMPLETE!")
        print("=" * 70)
        print(f"📂 Output directory: {self.output_dir}")
        print(f"🎬 Final video length: {len(frame_order)} frames")
        print(f"📉 Excluded frames: {len(frame_paths) - len(frame_order)}")
        
        return frame_order


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "frames")
    output_dir = os.path.join(project_root, "output", "reconstructed_frames_v3")
    
    reconstructor = ComprehensiveFrameReconstructor(frames_dir, output_dir)
    reconstructor.reconstruct()


if __name__ == "__main__":
    main()
