import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class TwoPhaseFrameReconstructor:
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
        Find TRUE starting point using multiple methods and vote.
        Combines: endpoint detection, flow direction, and sequence quality.
        """
        print("\n🔍 Phase 1: Finding TRUE starting point (multi-method voting)...")
        
        n_frames = len(features)
        similarity_matrix = cosine_similarity(features)
        
        # Method 1: Edge Detection - frames with asymmetric similarity
        print("   Method 1: Edge detection...")
        edge_scores = []
        for i in range(n_frames):
            sims = similarity_matrix[i].copy()
            sims[i] = 0  # Exclude self
            
            # Sort similarities
            sorted_sims = np.sort(sims)[::-1]
            
            # Edge frames have high similarity to one side, low to other
            # Calculate skewness: high similarities vs low similarities
            top_20_avg = sorted_sims[:20].mean()
            bottom_20_avg = sorted_sims[-20:].mean()
            edge_score = top_20_avg - bottom_20_avg
            edge_scores.append(edge_score)
        
        edge_scores = np.array(edge_scores)
        # Get top 20 edge candidates
        edge_candidates = set(np.argsort(edge_scores)[-20:].tolist())
        print(f"   Found {len(edge_candidates)} edge candidates")
        
        # Method 2: Low average similarity (typical of endpoints)
        print("   Method 2: Low average similarity...")
        avg_sims = similarity_matrix.mean(axis=1)
        low_sim_candidates = set(np.argsort(avg_sims)[:30].tolist())
        print(f"   Found {len(low_sim_candidates)} low-similarity candidates")
        
        # Method 3: Directional flow consistency
        print("   Method 3: Testing flow direction for all candidates...")
        combined_candidates = edge_candidates.union(low_sim_candidates)
        
        candidate_scores = {}
        
        for candidate in tqdm(list(combined_candidates), desc="   Evaluating candidates"):
            # Build a longer test sequence from this candidate
            visited = set([candidate])
            current = candidate
            sequence = [candidate]
            
            # Build 25-frame chain
            for _ in range(min(25, n_frames - 1)):
                sims = similarity_matrix[current].copy()
                sims[list(visited)] = -1
                next_idx = np.argmax(sims)
                if sims[next_idx] < 0.65:  # Lower threshold for longer chain
                    break
                sequence.append(next_idx)
                visited.add(next_idx)
                current = next_idx
            
            if len(sequence) < 10:
                continue
            
            # Test forward direction
            forward_flows = self.calculate_directional_flow(frame_paths, sequence)
            if not forward_flows:
                continue
            
            # Score based on:
            # 1. Flow consistency (low variance = good)
            # 2. Flow magnitude (should have clear direction)
            # 3. Sequence length (longer = better)
            flow_mean = np.mean(forward_flows)
            flow_std = np.std(forward_flows)
            flow_consistency = 1.0 / (flow_std + 0.1)
            flow_magnitude = abs(flow_mean)
            
            # Combined score
            score = (
                flow_consistency * 2.0 +      # Consistency is most important
                flow_magnitude * 1.0 +         # Clear direction
                len(sequence) * 0.1 +          # Longer sequence bonus
                edge_scores[candidate] * 0.5   # Edge score bonus
            )
            
            candidate_scores[candidate] = {
                'score': score,
                'sequence_length': len(sequence),
                'flow_mean': flow_mean,
                'flow_std': flow_std,
                'edge_score': edge_scores[candidate]
            }
        
        if not candidate_scores:
            print("   ⚠️  No valid candidates found, using fallback")
            return 0
        
        # Get top 5 candidates
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
        
        print(f"\n   Top 5 candidates:")
        for idx, (frame_idx, metrics) in enumerate(top_candidates, 1):
            print(f"   {idx}. Frame {frame_idx}: score={metrics['score']:.2f}, "
                  f"seq_len={metrics['sequence_length']}, "
                  f"flow_mean={metrics['flow_mean']:.3f}, "
                  f"flow_std={metrics['flow_std']:.3f}")
        
        # Method 4: For top candidates, test if reversing gives better flow
        best_start = None
        best_final_score = -float('inf')
        best_should_reverse = False
        
        print(f"\n   Testing forward vs backward for top candidates...")
        for candidate_idx, metrics in top_candidates[:3]:  # Test top 3
            # Build sequence from this candidate
            visited = set([candidate_idx])
            current = candidate_idx
            sequence = [candidate_idx]
            
            for _ in range(min(30, n_frames - 1)):
                sims = similarity_matrix[current].copy()
                sims[list(visited)] = -1
                next_idx = np.argmax(sims)
                if sims[next_idx] < 0.65:
                    break
                sequence.append(next_idx)
                visited.add(next_idx)
                current = next_idx
            
            # Test forward
            forward_flows = self.calculate_directional_flow(frame_paths, sequence)
            forward_consistency = 1.0 / (np.std(forward_flows) + 0.1) if forward_flows else 0
            forward_magnitude = abs(np.mean(forward_flows)) if forward_flows else 0
            
            # Test backward (reversed sequence)
            backward_flows = self.calculate_directional_flow(frame_paths, sequence[::-1])
            backward_consistency = 1.0 / (np.std(backward_flows) + 0.1) if backward_flows else 0
            backward_magnitude = abs(np.mean(backward_flows)) if backward_flows else 0
            
            # Score both directions
            forward_score = forward_consistency * 2.0 + forward_magnitude
            backward_score = backward_consistency * 2.0 + backward_magnitude
            
            if forward_score > backward_score and forward_score > best_final_score:
                best_final_score = forward_score
                best_start = candidate_idx
                best_should_reverse = False
            elif backward_score > forward_score and backward_score > best_final_score:
                best_final_score = backward_score
                best_start = sequence[-1]  # Start from end of sequence
                best_should_reverse = True
        
        if best_start is None:
            best_start = top_candidates[0][0]
            best_should_reverse = False
        
        direction = "forward" if not best_should_reverse else "backward (reversed)"
        print(f"\n   ✅ Selected start: Frame {best_start} (direction: {direction}, score: {best_final_score:.2f})")
        
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
        """Build main sequence using greedy algorithm"""
        print("\n🔄 Phase 2: Building main sequence (greedy)...")
        
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
        
        missing_frames = [i for i in range(n_frames) if not visited[i]]
        
        print(f"   ✅ Main sequence: {len(frame_order)} frames")
        print(f"   ⚠️  Missing frames: {len(missing_frames)}")
        
        return frame_order, missing_frames

    def insert_missing_frames(self, frame_order, missing_frames, similarity_matrix, neighborhood_size=30):
        """
        Insert missing frames using neighborhood-constrained search.
        
        Strategy:
        1. Find the best matching region in the sequence for each missing frame
        2. Search only within that neighborhood (±neighborhood_size positions)
        3. This keeps frames in their temporal locality
        
        Args:
            frame_order: Current ordered sequence
            missing_frames: List of frame indices not in sequence
            similarity_matrix: Full similarity matrix
            neighborhood_size: Search radius around best match position
        """
        if not missing_frames:
            print("\n✅ No missing frames to insert")
            return frame_order
        
        print(f"\n🔄 Phase 3: Inserting {len(missing_frames)} missing frames (neighborhood-constrained)...")
        print(f"   Neighborhood size: ±{neighborhood_size} positions")
        
        frame_order = list(frame_order)  # Make mutable
        
        for missing_idx in tqdm(missing_frames, desc="   Inserting frames"):
            # Step 1: Find the region where this frame belongs
            # Calculate average similarity to each position's neighborhood
            region_scores = []
            
            for pos in range(len(frame_order)):
                # Get frames in a window around this position
                window_start = max(0, pos - 5)
                window_end = min(len(frame_order), pos + 5)
                window_frames = frame_order[window_start:window_end]
                
                # Average similarity to this window
                window_sim = np.mean([similarity_matrix[missing_idx, f] for f in window_frames])
                region_scores.append(window_sim)
            
            # Find the best matching region
            if region_scores:
                best_region_pos = int(np.argmax(region_scores))
            else:
                best_region_pos = 0
            
            # Step 2: Search locally around the best region
            search_start = max(0, best_region_pos - neighborhood_size)
            search_end = min(len(frame_order) + 1, best_region_pos + neighborhood_size + 1)
            
            best_position = best_region_pos
            best_score = -float('inf')
            
            # Try inserting at positions within the neighborhood
            for insert_pos in range(search_start, search_end):
                # Calculate score based on similarity to immediate neighbors
                score = 0
                count = 0
                
                # Similarity to previous frame
                if insert_pos > 0:
                    prev_frame = frame_order[insert_pos - 1]
                    score += similarity_matrix[missing_idx, prev_frame]
                    count += 1
                
                # Similarity to next frame
                if insert_pos < len(frame_order):
                    next_frame = frame_order[insert_pos]
                    score += similarity_matrix[missing_idx, next_frame]
                    count += 1
                
                # Also consider neighbors at distance 2 (lookahead)
                if insert_pos > 1:
                    prev2_frame = frame_order[insert_pos - 2]
                    score += 0.5 * similarity_matrix[missing_idx, prev2_frame]
                    count += 0.5
                
                if insert_pos < len(frame_order) - 1:
                    next2_frame = frame_order[insert_pos + 1]
                    score += 0.5 * similarity_matrix[missing_idx, next2_frame]
                    count += 0.5
                
                # Average score
                if count > 0:
                    score /= count
                
                # Penalty for breaking a very strong connection
                if insert_pos > 0 and insert_pos < len(frame_order):
                    prev_frame = frame_order[insert_pos - 1]
                    next_frame = frame_order[insert_pos]
                    existing_pair_sim = similarity_matrix[prev_frame, next_frame]
                    
                    # Strong penalty for breaking excellent connections
                    if existing_pair_sim > 0.95:
                        score -= 0.3
                    elif existing_pair_sim > 0.90:
                        score -= 0.15
                
                # Bonus for maintaining flow consistency
                # Check if inserting here maintains similar similarity levels
                if insert_pos > 0 and insert_pos < len(frame_order):
                    prev_frame = frame_order[insert_pos - 1]
                    next_frame = frame_order[insert_pos]
                    left_sim = similarity_matrix[missing_idx, prev_frame]
                    right_sim = similarity_matrix[missing_idx, next_frame]
                    existing_sim = similarity_matrix[prev_frame, next_frame]
                    
                    # Bonus if we maintain similar similarity levels (smooth transition)
                    if min(left_sim, right_sim) > existing_sim * 0.8:
                        score += 0.1
                
                if score > best_score:
                    best_score = score
                    best_position = insert_pos
            
            # Insert at best position within neighborhood
            frame_order.insert(best_position, missing_idx)
        
        print(f"   ✅ All frames inserted. Final sequence: {len(frame_order)} frames")
        return frame_order

    def verify_and_reverse_if_needed(self, frame_paths, frame_order):
        """Check if sequence should be reversed"""
        print("\n🔄 Phase 4: Verifying direction...")
        
        n = len(frame_order)
        if n < 2:
            return frame_order
        
        # Sample from middle 50%
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
            print(f"   🔄 Reversing sequence")
            return frame_order[::-1]
        
        print(f"   ✅ Direction confirmed")
        return frame_order

    def local_refinement_2opt(self, frame_order, similarity_matrix, iterations=3):
        """
        Apply 2-opt local search to fix small ordering errors.
        This can swap segments to improve local coherence.
        """
        print(f"\n🔄 Phase 5: Local refinement (2-opt, {iterations} iterations)...")
        
        n = len(frame_order)
        current_order = list(frame_order)
        
        def calculate_cost(order):
            cost = 0
            for i in range(len(order) - 1):
                cost += similarity_matrix[order[i], order[i+1]]
            return cost
        
        initial_cost = calculate_cost(current_order)
        print(f"   Initial cost: {initial_cost:.4f}")
        
        for iteration in range(iterations):
            improved = False
            
            # Try swapping segments
            for i in range(1, n - 2):
                for k in range(i + 1, min(i + 20, n - 1)):  # Limit range for speed
                    # Reverse segment [i:k+1]
                    new_order = current_order[:i] + current_order[i:k+1][::-1] + current_order[k+1:]
                    new_cost = calculate_cost(new_order)
                    
                    if new_cost > initial_cost:
                        current_order = new_order
                        initial_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                print(f"   Iteration {iteration+1}: Cost improved to {initial_cost:.4f}")
            else:
                print(f"   Iteration {iteration+1}: No improvement found")
                break
        
        print(f"   ✅ Final cost: {initial_cost:.4f}")
        return current_order

    def reconstruct(self):
        """Main two-phase reconstruction pipeline"""
        print("=" * 70)
        print("TWO-PHASE FRAME RECONSTRUCTION (V5.2)")
        print("Improved Start Detection + Neighborhood-Constrained Insertion")
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
        
        # Compute similarity matrix once
        print("\n🔄 Computing similarity matrix...")
        similarity_matrix = cosine_similarity(features)
        
        # Phase 1: Find optimal start
        start_idx = self.find_true_start_with_flow(frame_paths, features)
        
        # Phase 2: Build main sequence
        frame_order, missing_frames = self.greedy_order_frames(features, start_idx)
        
        # Phase 3: Insert missing frames (with neighborhood constraint)
        frame_order = self.insert_missing_frames(frame_order, missing_frames, similarity_matrix, 
                                                 neighborhood_size=30)
        
        # Phase 4: Verify direction
        frame_order = self.verify_and_reverse_if_needed(frame_paths, frame_order)
        
        # Phase 5: Local refinement
        frame_order = self.local_refinement_2opt(frame_order, similarity_matrix, iterations=3)
        
        print(f"\n📊 Final sequence length: {len(frame_order)} frames")
        print(f"   Expected duration at 30fps: {len(frame_order)/30:.2f} seconds")
        
        # Verify all frames are included
        if len(set(frame_order)) != len(frame_paths):
            print(f"   ⚠️  WARNING: Some frames duplicated or missing!")
        else:
            print(f"   ✅ All {len(frame_paths)} frames included (no duplicates)")
        
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
            order_path = os.path.join(self.output_dir, "frame_order_v5.txt")
            np.savetxt(order_path, np.array(frame_order, dtype=int), fmt="%d")
            print(f"📝 Frame order saved to: {order_path}")
        
        print("\n" + "=" * 70)
        print("✅ RECONSTRUCTION COMPLETE!")
        print("=" * 70)
        print(f"📂 Output: {self.output_dir}")
        print(f"🎬 Final video: {len(frame_order)} frames = {len(frame_order)/30:.2f} seconds")
        
        return frame_order


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    frames_dir = os.path.join(project_root, "frames")
    output_dir = os.path.join(project_root, "output", "reconstructed_frames_v5_2")
    
    reconstructor = TwoPhaseFrameReconstructor(frames_dir, output_dir)
    reconstructor.reconstruct()


if __name__ == "__main__":
    main()
