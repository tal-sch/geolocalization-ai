import torch
import torch.nn as nn

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

    def unfreeze_step(self, blocks_per_step=2):
        """
        Smart Unfreezing:
        1. Checks what is currently frozen.
        2. Unfreezes the next 'blocks_per_step' layers.
        3. Returns True if successful, False if the whole model is already open.
        """
        #todo: refactor to avoid code duplication with training script
        
        # check if the first block is already unfrozen - all blocks are unfrozen
        if next(self.backbone.blocks[0].parameters()).requires_grad:
            return False 

        # find the first frozen block from the end
        first_frozen_idx = -1
        for i in range(11, -1, -1):
            if not next(self.backbone.blocks[i].parameters()).requires_grad:
                first_frozen_idx = i
                break
        
        if first_frozen_idx == -1: # all blocks are unfrozen
            return False 

        # calculate block indices to unfreeze
        target_idx = max(0, first_frozen_idx - blocks_per_step + 1)

        print(f"[{type(self).__name__}] Unfreezing Blocks {first_frozen_idx} -> {target_idx}...")
        
        for i in range(first_frozen_idx, target_idx - 1, -1):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True
        
        return True