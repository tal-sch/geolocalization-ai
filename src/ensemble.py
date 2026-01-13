# ============================================================================
# ENSEMBLE WRAPPER
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
import numpy as np
from src.models import GeoCLIP


class GeoCLIPEnsemble:
    def __init__(self, model_configs, device='cuda'):
        self.models = []
        self.device = device
        
        print(f"Initializing Ensemble with {len(model_configs)} models...")
        
        for config in model_configs:
            path = config['path']
            name = config['name']
            
            # FIXED: Removed 'use_attention' and 'use_multitask'
            # to match your Simple GeoCLIP class
            model = GeoCLIP(
                dropout_rate=0.0, # Dropout 0 for inference
                model_name=name
            ).to(device)
            
            try:
                # Load weights
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                self.models.append(model)
                print(f" ✅ Loaded {name}")
            except Exception as e:
                print(f" ❌ Failed to load {path}: {e}")
    
    def predict(self, x):
        """Average predictions from all models"""
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred.cpu().numpy())
        return torch.tensor(np.mean(predictions, axis=0))

    def predict_with_tta(self, x, num_augments=5):
        """Test-Time Augmentation + Ensemble"""
        import torchvision.transforms as T
        
        all_predictions = []
        
        # 1. Prediction on clean image
        all_predictions.append(self.predict(x).numpy())
        
        # 2. Augmented predictions
        # Simple augmentations: shift slightly, change brightness
        tta_transform = T.Compose([
            T.RandomAffine(degrees=0, translate=(0.02, 0.02)), # Small shift
            T.ColorJitter(brightness=0.1, contrast=0.1)        # Small light change
        ])
        
        for _ in range(num_augments - 1):
            aug_batch = torch.stack([tta_transform(img) for img in x])
            all_predictions.append(self.predict(aug_batch).numpy())
            
        return torch.tensor(np.median(all_predictions, axis=0))

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
