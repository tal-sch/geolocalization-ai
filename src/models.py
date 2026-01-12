import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(r"D:\programming_projects\university\geolocalization-ai\LightGlue") # git clone https://github.com/cvg/LightGlue.git
from lightglue import LightGlue, SuperPoint

import matplotlib.pyplot as plt
import contextily as cx

class ConvNet(nn.Module):
    def __init__(self, use_dropout=False, use_dropout2d=False):
        super(ConvNet, self).__init__()

        self.use_dropout = use_dropout
        self.use_dropout2d = use_dropout2d

        self.conv1 = nn.Conv2d(3, 128, 5, padding="same")
        self.conv2 = nn.Conv2d(128, 64, 3, padding="same")

        if self.use_dropout2d:
            self.spatial_dropout = nn.Dropout2d(p=0.2)

        # Global Average Pooling ensures this stays 64 regardless of image size
        self.fc1 = nn.Linear(64, 256)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)

        # 2. Vector of 2 for [Latitude, Longitude]
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.conv2(x)

        if self.use_dropout2d:
            x = self.spatial_dropout(x)

        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)  # Global Avg Pool
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc2(x)  # Linear output for regression
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # 'p' is a learnable parameter.
        # p=1 is Average Pooling, p=Infinity is Max Pooling.
        # The model will learn the best value between them.
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # The GeM formula
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class ConvNet2(nn.Module):
    def __init__(self, use_dropout=True):
        super(ConvNet2, self).__init__()

        # Layer 1: 3 -> 32 filters
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Stabilizes learning

        # Layer 2: 32 -> 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Layer 3: 64 -> 128 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Layer 4: 128 -> 256 filters
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 2)  # [Latitude, Longitude]

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 256 -> 128

        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 128 -> 64

        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 64 -> 32

        # Conv Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, 1)  # Global Average Pooling

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConvNet3(nn.Module):
    def __init__(self, use_dropout=False, use_dropout2d=False):
        super(ConvNet3, self).__init__()

        # --- Block 1 ---
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),  # Swish
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Block 2 ---
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

        # --- Block 3 ---
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

        # --- Block 4 ---
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

        # --- Block 5 ---
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

        # --- Global Pooling ---
        # The key upgrade for Geolocalization
        self.global_pool = GeM()

        # --- Dropout ---
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.dropout2d = nn.Dropout2d(0.5) if use_dropout2d else None

        # --- Regression Head ---
        self.regressor = nn.Sequential(
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.dropout2d:
            out = self.dropout2d(out)

        out = self.layer5(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.regressor(out)
        return out

class MultiTaskDINOGeo(nn.Module):
    def __init__(self, num_zones):
        super().__init__()
        # Load Pre-trained DINOv2 Model
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # unfreeze the last block
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True

        # Keep final layernorm unfrozen
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        # Shared "Reasoning" Layer
        self.shared = nn.Sequential(
            nn.Linear(384, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # HEAD 1: Regression (Lat/Lon)
        self.reg_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

        # HEAD 2: Classification (Zone ID)
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_zones) # Output: Probability for each zone
        )

    def forward(self, x):
        features = self.backbone(x)
        shared_feat = self.shared(features)
        
        coords = self.reg_head(shared_feat)
        zones = self.cls_head(shared_feat)
        return coords, zones
    
def gated_contrastive_loss(embeddings, backbone_features, real_coords, margin=0.2):
    """
    student_embeddings: (Batch, 128) - The vector we are training
    frozen_features:    (Batch, 384) - The raw DINO output (The "Truth" about similarity)
    real_coords:        (Batch, 2)   - Normalized GPS
    """
    
    # 1. Calculate GPS Distances (Physical)
    gps_dist = torch.cdist(real_coords, real_coords)
    # Define Physical Neighbors (e.g., < 10-15m). 
    # 0.005 is roughly 0.5% of map width. Tune this if needed!
    is_gps_neighbor = (gps_dist < 0.005).float()
    
    # 2. Calculate Visual Distances (Semantic Gate)
    # Use the frozen features to decide if they "look" the same
    frozen_norm = nn.functional.normalize(backbone_features, p=2, dim=1)
    visual_dist = torch.cdist(frozen_norm, frozen_norm)
    
    # 0.5 is a standard threshold for Cosine Similarity distance
    is_visual_neighbor = (visual_dist < 0.5).float()
    
    # 3. Create the "Gate"
    # ONLY pull if physically close AND visually similar (e.g. both facing North)
    is_positive_pair = is_gps_neighbor * is_visual_neighbor
    
    # 4. Calculate Student Embedding Distances (The ones we are training)
    student_dist = torch.cdist(embeddings, embeddings)
    
    # 5. The Loss Calculation
    # POSITIVE: Pull 'True' matches together
    loss_pos = is_positive_pair * torch.pow(student_dist, 2)
    
    # NEGATIVE: Push 'Far' matches apart
    # Note: We push apart anything that is physically far (regardless of visual similarity)
    is_negative_pair = (1 - is_gps_neighbor)
    loss_neg = is_negative_pair * torch.pow(torch.relu(margin - student_dist), 2)
    
    # Mask out self-comparisons (diagonal)
    mask = torch.triu(torch.ones_like(gps_dist), diagonal=1)
    
    final_loss = (loss_pos + loss_neg) * mask
    
    # Normalize by non-zero pairs to avoid instability
    return final_loss.sum() / (mask.sum() + 1e-6)

class MetricDINOGeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Freeze most of the Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze ONLY the last block and Norm layer
        # This allows adaptation to campus textures without losing "world knowledge"
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
            
        # 2. The Projector (The "Fingerprint Maker")
        # Compresses 384 dims -> 128 dims. 
        # This vector is what we will use for k-NN search.
        self.projector = nn.Sequential(
            nn.Linear(384, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 128) 
        )

        # 3. Auxiliary Regression Head (The "Compass")
        # We keep this ONLY to help the model learn global placement during training.
        # We will IGNORE this output during final inference.
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2) 
        )

    def forward(self, x):
        # 1. Get Raw Features (We need these for the Loss Gate)
        backbone_features = self.backbone(x)
        
        # 2. Create Embedding
        embedding = self.projector(backbone_features)
        # Normalize to unit sphere (Critical for Contrastive Loss!)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        # 3. Predict Coords (Auxiliary)
        pred_coords = self.reg_head(embedding)
        
        return embedding, pred_coords, backbone_features
    

class HierarchicalLocalizer:
    def __init__(self, train_dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = train_dataset
        
        print(f"Loading models on {self.device}...")
        
        # 1. Global Features (DINOv2) - Good at "What does it look like?"
        self.global_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.global_model.eval()
        self.dino_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 2. Local Features (LightGlue) - Good at "Do the shapes match?"
        # Increased keypoints to 2048 for better detection in dark/complex scenes
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.to_grayscale = T.Grayscale() # superpoint works best on grayscale images

        # 3. Build Index
        self.db_descriptors, self.db_local_feats, self.db_gps = self.build_database_index()

    def build_database_index(self):
        loader = DataLoader(self.train_dataset, batch_size=32, shuffle=False, num_workers=0)

        global_descs = []
        local_feats_cache = []
        gps_coords = []
        
        with torch.no_grad():
            for images, coords in tqdm(loader, desc="Building Index"):
                images = images.to(self.device, non_blocking=True)

                # run DINO to get global descriptors 
                dino_imgs = self.dino_norm(images)
                g_desc = self.global_model(dino_imgs)
                g_desc = g_desc / g_desc.norm(dim=-1, keepdim=True)
                global_descs.append(g_desc.cpu())

                # run SuperPoint to get local features
                gray_imgs = self.to_grayscale(images)
                
                # Iterate through the batch manually
                for i in range(gray_imgs.shape[0]):
                    # Get single image (1, H, W) -> unsqueeze to (1, 1, H, W)
                    single_img = gray_imgs[i].unsqueeze(0)
                    
                    # Extract features
                    feats = self.extractor.extract(single_img)
                    
                    # Move to CPU to save VRAM
                    # LightGlue returns tensors with batch dim 1, which is what we want to store
                    feats_cpu = {k: v.cpu() for k, v in feats.items()}
                    local_feats_cache.append(feats_cpu)
                                
                gps_coords.append(coords)

        return torch.cat(global_descs), local_feats_cache, torch.cat(gps_coords)

    def predict(self, query_img, top_k_dino=20, top_k_lightglue=5, MIN_INLIER_THRESHOLD = 100, debug=False, true_coords=None):
        """
        query_img: A single [C, H, W] tensor in [0, 1] range (RGB).
        """
        if query_img.dim() == 3:
            query_img = query_img.unsqueeze(0) # Add batch dim
        
        query_img = query_img.to(self.device)

        with torch.no_grad():
            # extract global descriptor and normalize
            dino_in = self.dino_norm(query_img)
            query_vec = self.global_model(dino_in)
            query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
            
            # Compute similarities and retrieve top-k
            sims = query_vec @ self.db_descriptors.to(self.device).t()
            scores, indices = torch.topk(sims.squeeze(), top_k_dino)
            indices = indices.cpu().numpy()

        best_idx = indices[0] # Default to best visual match

        with torch.no_grad():
            gray_query = self.to_grayscale(query_img)
            feats_query = self.extractor.extract(gray_query)

        valid_matches = []

        for idx in indices:
            # retrieve pre-computed local features of top-k candidates
            feats_cand = self.db_local_feats[idx]
            
            # Move candidate features to GPU for matching
            feats_cand_gpu = {k: v.to(self.device) for k, v in feats_cand.items()}

            # LightGlue Matcher
            matches = self.matcher({'image0': feats_query, 'image1': feats_cand_gpu})
            
            # Count robust matches (inliers)
            pruned = matches['matches'][0] # [0] because batch=1
            inliers = len(pruned)

            if inliers > MIN_INLIER_THRESHOLD:
                gps_cand = self.db_gps[idx].numpy()
                valid_matches.append((gps_cand, inliers, idx))

        if not valid_matches:
            # No good matches found, return best visual match by dino
            return self.db_gps[best_idx].numpy()
        
        valid_matches.sort(key=lambda x: x[1], reverse=True)
        top_k_lightglue_matches = valid_matches[:top_k_lightglue]
        
        total_weight = sum(m[1] for m in top_k_lightglue_matches) # Sum of inlier counts
        
        weighted_lat = sum(m[0][0] * m[1] for m in top_k_lightglue_matches) / total_weight # Latitude
        weighted_lon = sum(m[0][1] * m[1] for m in top_k_lightglue_matches) / total_weight # Longitude
        
        if debug and true_coords is not None:
            self.visualize_matches(query_img, top_k_lightglue_matches, (weighted_lat, weighted_lon), true_coords)
        
        return np.array([weighted_lat, weighted_lon], dtype=np.float32)

    def validate(self, val_loader, top_k_dino=20, top_k_lightglue=5, inlier_threshold=100, debug=False):
            """
            Efficient validation loop using Weighted Interpolation.
            
            Args:
                top_k_dino: How many candidates to retrieve globally.
                top_k_lightglue: How many best matches to use for interpolation.
                inlier_threshold: Min matches to accept a candidate for interpolation.
            """
            print(f"\n--- Starting Validation on {len(val_loader.dataset)} images ---")
            
            # 1. Set Evaluation Mode
            self.global_model.eval()
            self.extractor.eval()
            self.matcher.eval()
            
            # 2. Pre-load Database Descriptors to GPU (Optimization)
            # We do this ONCE, not every iteration
            db_desc_gpu = self.db_descriptors.to(self.device)
            
            results = [] # Stores (Error_Meters, Pred_Lat, Pred_Lon, Gt_Lat, Gt_Lon)
            
            # Haversine Distance Helper
            def haversine_np(lat1, lon1, lat2, lon2):
                R = 6371000
                phi1, phi2 = np.radians(lat1), np.radians(lat2)
                dphi = np.radians(lat2 - lat1)
                dlambda = np.radians(lon2 - lon1)
                a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                return R * c

            pbar = tqdm(val_loader, desc="Validating", unit="batch")
            with torch.no_grad():
                for images, gt_coords in pbar:
                    images = images.to(self.device)
                    batch_size = images.shape[0]
                    
                    # --- A. Global Search (Batched for Speed) ---
                    dino_in = self.dino_norm(images)
                    query_vecs = self.global_model(dino_in)
                    query_vecs = query_vecs / query_vecs.norm(dim=-1, keepdim=True)
                    
                    # Matrix Mult: [Batch, Dim] @ [Dim, DB_Size] -> [Batch, DB_Size]
                    sims = query_vecs @ db_desc_gpu.t()
                    topk_scores, topk_indices = torch.topk(sims, k=top_k_dino)
                    topk_indices = topk_indices.cpu().numpy()
                    
                    # --- B. Geometric Verification (Per Query) ---
                    for b in range(batch_size):
                        # 1. Extract Local Features for Query ONCE
                        q_img = images[b].unsqueeze(0) # [1, C, H, W]
                        gray_q = self.to_grayscale(q_img)
                        feats_q = self.extractor.extract(gray_q) # Cached on GPU
                        
                        candidates = topk_indices[b]
                        valid_matches = []
                        
                        # 2. Check Candidates
                        for db_idx in candidates:
                            # Fetch DB features (move to GPU on demand)
                            feats_db = {k: v.to(self.device) for k, v in self.db_local_feats[db_idx].items()}
                            
                            # Match
                            matches = self.matcher({'image0': feats_q, 'image1': feats_db})
                            inliers = len(matches['matches'][0])
                            
                            if inliers >= inlier_threshold:
                                gps_cand = self.db_gps[db_idx].numpy()
                                valid_matches.append((gps_cand, inliers))
                        
                        # --- C. Weighted Interpolation ---
                        if not valid_matches:
                            # Fallback: Top-1 DINO match
                            pred_gps = self.db_gps[candidates[0]].numpy()
                        else:
                            # Sort by inliers (descending)
                            valid_matches.sort(key=lambda x: x[1], reverse=True)
                            
                            # Take top N best geometric matches
                            top_n = valid_matches[:top_k_lightglue]
                            
                            # Weighted Average
                            total_w = sum(m[1] for m in top_n)
                            w_lat = sum(m[0][0] * m[1] for m in top_n) / total_w
                            w_lon = sum(m[0][1] * m[1] for m in top_n) / total_w
                            pred_gps = np.array([w_lat, w_lon])
                        
                        # --- D. Calculate Error ---
                        gt = gt_coords[b].numpy()
                        err = haversine_np(pred_gps[0], pred_gps[1], gt[0], gt[1])
                        results.append((err, pred_gps[0], pred_gps[1], gt[0], gt[1]))
                        pbar.set_postfix({'Last Err (m)': f"{err:.2f}",
                                           'Avg Err (m)': f"{np.mean([r[0] for r in results]):.2f}",
                                           'Med Err (m)': f"{np.median([r[0] for r in results]):.2f}",
                                           'Max Err (m)': f"{np.max([r[0] for r in results]):.2f}"})
                        if debug:
                            self.visualize_matches(q_img, top_n, pred_gps, gt)

                        

            # --- Report Statistics ---
            errors = np.array([r[0] for r in results])
            print(f"\nResults Summary:")
            print(f"Mean Error:   {np.mean(errors):.2f}m")
            print(f"Median Error: {np.median(errors):.2f}m")
            print(f"Max Error:    {np.max(errors):.2f}m")
            print(f"Accuracy < 5m:  {np.mean(errors < 5.0) * 100:.1f}%")
            print(f"Accuracy < 10m: {np.mean(errors < 10.0) * 100:.1f}%")
            print(f"Accuracy < 25m: {np.mean(errors < 25.0) * 100:.1f}%")
            
            return results

    def visualize_matches(self, query_img, top_k_matches, weighted_coords, true_coords=None):
        if not top_k_matches:
            return

        weighted_lat, weighted_lon = weighted_coords

        # Plot the original image and the top N matches
        fig, axes = plt.subplots(1, len(top_k_matches) + 1, figsize=(15, 5))
        
        # Plot the query image
        img_to_plot = query_img.squeeze(0) if query_img.dim() == 4 else query_img
        axes[0].imshow(img_to_plot.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Query Image")
        axes[0].axis("off")

        # Plot the top N matched images
        for i, (gps, count, idx) in enumerate(top_k_matches):
            matched_image = self.train_dataset[idx][0]  # Retrieve the image from the dataset
            axes[i + 1].imshow(matched_image.permute(1, 2, 0).cpu().numpy())
            axes[i + 1].set_title(f"({i + 1}) Inliers: {count}")
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.show()

        # Plot the top k LightGlue match coordinates, weighted average, and true coordinate
        fig, ax = plt.subplots(figsize=(8, 3))

        # Extract top k LightGlue match coordinates
        match_coords = np.array([m[0] for m in top_k_matches])
        match_inliers = [m[1] for m in top_k_matches]

        if true_coords is not None:
            # Plot true coordinate (Green)
            ax.scatter(true_coords[1], true_coords[0], c='lime', label='True', alpha=0.8, s=50, edgecolors='black', zorder=3)

        # Plot weighted average coordinate (Red)
        ax.scatter(weighted_lon, weighted_lat, c='red', label='Weighted Avg', alpha=0.8, s=50, edgecolors='black', zorder=3)

        # Plot top k LightGlue match coordinates 
        for coord, inlier_count in zip(match_coords, match_inliers):
            ax.scatter(coord[1], coord[0], c='blue', alpha=0.6, s=50, edgecolors='black', zorder=2, label='Matches' if 'Matches' not in ax.get_legend_handles_labels()[1] else "")
            ax.text(coord[1], coord[0], f'{inlier_count}', fontsize=8, ha='center', va='center', color='black', zorder=3)

        ax.set_axis_off()
        ax.set_aspect('equal')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax.margins(0, 0)

        try:
            cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik, alpha=1.0, reset_extent=False, zoom=19)
        except Exception as e:
            print(f"Could not fetch map tiles: {e}")

        ax.legend(loc='upper right')
        plt.show()