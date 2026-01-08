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
    """
    Ensemble of multiple GeoCLIP models for improved predictions
    """
    def __init__(self, model_paths, device='cuda'):
        self.models = []
        self.device = device
        
        for path in model_paths:
            model = GeoCLIP(
                dropout_rate=0.4,
                model_name="openai/clip-vit-large-patch14",
                use_attention=True,
                use_multitask=False  # Disable for inference
            ).to(device)
            
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
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
        
        # Original image
        all_predictions.append(self.predict(x).numpy())
        
        # Augmented versions
        tta_transform = T.Compose([
            T.RandomAffine(degrees=5, translate=(0.03, 0.03)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
        
        for _ in range(num_augments - 1):
            # Apply augmentation to each image in batch
            aug_batch = torch.stack([tta_transform(img) for img in x])
            all_predictions.append(self.predict(aug_batch).numpy())
        
        return torch.tensor(np.mean(all_predictions, axis=0))


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
