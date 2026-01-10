import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T

import os
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd

import sys
sys.path.append(r"D:\programming_projects\university\geolocalization-ai\LightGlue") # git clone https://github.com/cvg/LightGlue.git
from lightglue import LightGlue, SuperPoint



class HierarchicalLocalizer:
    def __init__(self, train_dataset, cache_path="full_dataset_index_cache.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = train_dataset
        self.cache_path = cache_path
        
        print(f"Loading models on {self.device}...")
        
        # 1. Global Features (DINOv2) - Good at "What does it look like?"
        self.global_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.global_model.eval()
        self.dino_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 2. Local Features (LightGlue) - Good at "Do the shapes match?"
        # Increased keypoints to 4096 for better detection in dark/complex scenes
        self.extractor = SuperPoint(max_num_keypoints=4096).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

        self.to_grayscale = T.Grayscale() # superpoint works best on grayscale images

        if os.path.exists(self.cache_path):
            print(f"Loading cached index from {self.cache_path}...")
            self.load_index()
        else:
            print("No cache found. Building index...")
            self.db_descriptors, self.db_local_feats, self.db_gps = self.build_database_index()
            self.save_index()

        self.db_descriptors = self.db_descriptors.to(self.device)

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
    

    def predict(self, query_img, top_k=20, top_n_matches=5, MIN_INLIER_THRESHOLD = 100, debug=False):
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
            scores, indices = torch.topk(sims.squeeze(), top_k)
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
        top_n_matches = valid_matches[:top_n_matches]
        
        total_weight = sum(m[1] for m in top_n_matches) # Sum of inlier counts
        
        weighted_lat = sum(m[0][0] * m[1] for m in top_n_matches) / total_weight # Latitude
        weighted_lon = sum(m[0][1] * m[1] for m in top_n_matches) / total_weight # Longitude
        
        if debug:
            # Plot the original image and the top N matches
            fig, axes = plt.subplots(1, len(top_n_matches) + 1, figsize=(15, 5))

            # Plot the query image
            axes[0].imshow(query_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
            axes[0].set_title("Query Image")
            axes[0].axis("off")

            # Plot the top N matched images
            for i, (gps, count, idx) in enumerate(top_n_matches):
                matched_image = self.train_dataset[idx][0]  # Retrieve the image from the dataset
                axes[i + 1].imshow(matched_image.permute(1, 2, 0).cpu().numpy())
                axes[i + 1].set_title(f"({i + 1}) Inliers: {count}")
                axes[i + 1].axis("off")

            plt.tight_layout()
            plt.show()
        
        return np.array([weighted_lat, weighted_lon], dtype=np.float32)
    
    def save_index(self):
        print(f"Saving index to {self.cache_path}")
        torch.save({
            'global': self.db_descriptors.cpu(),
            'local': self.db_local_feats,
            'gps': self.db_gps
        }, self.cache_path)
        

    def load_index(self):
        data = torch.load(self.cache_path)
        self.db_descriptors = data['global']
        self.db_local_feats = data['local']
        self.db_gps = data['gps']

    def _haversine_numpy(self, lat1, lon1, lat2, lon2):
        R = 6371000.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))