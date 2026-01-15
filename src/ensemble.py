# ============================================================================
# ENSEMBLE WRAPPER
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
import numpy as np
from src.models import GeoCLIP


import torch
import numpy as np
from src.models import GeoCLIP, GeoDINO, PretrainedResNet  # Ensure these are imported

class GeoEnsemble:
    def __init__(self, model_configs, device, weights=None):
        self.models = []
        self.device = device
        
        # Handle Weights
        if weights is not None:
            # Normalize just in case
            weights = np.array(weights)
            weights = weights / weights.sum()
            self.weights = torch.tensor(weights, device=device, dtype=torch.float32)
            print(f"⚖️ Ensemble Weights Set: {self.weights.cpu().numpy()}")
        else:
            self.weights = None
            print("⚖️ Ensemble Weights: None (Equal Voting)")

        print(f"🧩 Initializing Ensemble with {len(model_configs)} models...")
        
        for config in model_configs:
            m_type = config['type']
            m_name = config['name']
            m_path = config['path']
            # Get dropout (Critical for correct architecture init)
            m_drop = config.get('dropout', 0.5) 
            
            # 1. Instantiate
            if m_type == "CLIP":
                model = GeoCLIP(dropout_rate=m_drop, model_name=m_name)
            elif m_type == "DINO":
                model = GeoDINO(dropout_rate=m_drop, model_name=m_name)
            
            # 2. Load Weights
            print(f"   -> Loading {m_type} (Drop={m_drop}) from {m_path}...")
            state_dict = torch.load(m_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # 3. Optimize
            model.to(device)
            model.eval()
            if "RTX" in torch.cuda.get_device_name(0):
                model = model.to(memory_format=torch.channels_last)
                
            self.models.append(model)
            
    def predict(self, images):
        if images.device != self.device:
            images = images.to(self.device)
            
        with torch.no_grad():
            # 1. Get Individual Predictions
            # We assume Model 0 is CLIP, Model 1 is DINO (based on your config order)
            p_clip = self.models[0](images) 
            p_dino = self.models[1](images) 

            # 2. Calculate Disagreement (Euclidean Distance in Normalized Space)
            # (Batch_Size,) tensor of distances
            disagreement = torch.norm(p_clip - p_dino, dim=1)
            
            # 3. Define Threshold (0.15 normalized is roughly ~200-300m in real world depending on scale)
            # You might need to tune this. Start with 0.15.
            veto_threshold = 0.15 
            
            # 4. Create Decision Mask
            # 1.0 if they disagree (Trust DINO only), 0.0 if they agree (Average)
            mask = (disagreement > veto_threshold).float().unsqueeze(1)
            
            # 5. Calculate Standard Weighted Average
            stacked = torch.stack([p_clip, p_dino])
            if self.weights is None:
                weighted_avg = torch.mean(stacked, dim=0)
            else:
                w = self.weights.view(-1, 1, 1) 
                weighted_avg = (stacked * w).sum(dim=0)

            # 6. Final Decision: 
            # If mask is 1 (High Disagreement) -> Take p_dino
            # If mask is 0 (Low Disagreement)  -> Take weighted_avg
            final_pred = (mask * p_dino) + ((1 - mask) * weighted_avg)
            
            return final_pred

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_region_labels(coordinates, bounds):
    """
    Convert lat/lon to quadrant labels (NW=0, NE=1, SW=2, SE=3)
    
    Args:
        coordinates: (N, 2) array of [lat, lon]
        bounds: dict with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
    
    Returns:
        labels: (N,) array of region indices
    """
    lat_mid = (bounds['lat_min'] + bounds['lat_max']) / 2
    lon_mid = (bounds['lon_min'] + bounds['lon_max']) / 2
    
    labels = []
    for lat, lon in coordinates:
        if lat >= lat_mid and lon < lon_mid:
            labels.append(0)  # NW
        elif lat >= lat_mid and lon >= lon_mid:
            labels.append(1)  # NE
        elif lat < lat_mid and lon < lon_mid:
            labels.append(2)  # SW
        else:
            labels.append(3)  # SE
    
    return np.array(labels)


def multitask_loss(coord_pred, coord_true, region_pred, region_true, 
                   coord_weight=1.0, region_weight=0.3):
    """
    Combined loss for multi-task learning
    
    Args:
        coord_pred: (Batch, 2) predicted coordinates
        coord_true: (Batch, 2) true coordinates
        region_pred: (Batch, num_regions) region logits
        region_true: (Batch,) region labels
        coord_weight: weight for coordinate loss
        region_weight: weight for region loss
    """
    coord_loss = F.smooth_l1_loss(coord_pred, coord_true, beta=1.0)
    region_loss = F.cross_entropy(region_pred, region_true)
    
    total_loss = coord_weight * coord_loss + region_weight * region_loss
    
    return total_loss, coord_loss, region_loss
