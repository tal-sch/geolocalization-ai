import torch
import torch.nn as nn
import torch.nn.functional as F


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