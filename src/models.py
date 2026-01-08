import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPVisionModel


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


class PretrainedResNet(nn.Module):
    def __init__(self, use_dropout=True):
        super(PretrainedResNet, self).__init__()
        
        # 1. Load the Pretrained Beast
        # "DEFAULT" downloads the best available weights (ImageNet)
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        
        # 2. (Optional) Freeze early layers
        # This prevents the model from forgetting "basic vision" during the first few epochs.
        # We freeze everything initially.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Replace the Standard Pooling with GeM (The SOTA upgrade)
        # ResNet's "avgpool" is usually just an AdaptiveAvgPool2d
        self.backbone.avgpool = GeM() 
        
        # 4. Replace the Classification Head (fc)
        # ResNet50 outputs 2048 features before the final layer
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 1024),
            nn.SiLU(),             # Modern Activation
            nn.Dropout(0.7 if use_dropout else 0.0),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.7 if use_dropout else 0.0),
            nn.Linear(512, 2)      # [Latitude, Longitude]
        )

    def forward(self, x):
        return self.backbone(x)
        
    def unfreeze(self):
        """Call this method later to unlock the whole model for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # Unfreeze only last ResNet block
    def unfreeze_layer4(self):
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

    # Optional deeper fine-tuning
    def unfreeze_layer3(self):
        for p in self.backbone.layer3.parameters():
            p.requires_grad = True



class GeoDINOv2(nn.Module):
    def __init__(self, dropout_rate=0.7): # High dropout for the head
        super(GeoDINOv2, self).__init__()
        
        # 1. Load DINOv2 Small (Lightweight, massive intelligence)
        # We load from Torch Hub. It downloads automatically.
        print("Loading DINOv2-Small...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # 2. FREEZE THE BACKBONE (Critical for small datasets)
        # We trust DINO's features more than we trust our training loop.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. GeM Pooling 
        # DINOv2-Small outputs 384 dimensions.
        self.pooling = GeM(p=3.0)
        
        # 4. Regression Head
        self.head = nn.Sequential(
            nn.Linear(384, 1024),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2) # Lat/Lon
        )

    def forward(self, x):
        # DINOv2 requires image sides to be multiples of 14.
        # Ensure your dataset resizes to (224, 294) or (224, 224).
        
        # Forward pass through backbone
        # We want the "patch tokens", not the CLS token.
        features = self.backbone.forward_features(x)
        patch_tokens = features['x_norm_patchtokens'] # Shape: (Batch, N_Patches, 384)
        
        # Reshape for GeM: (Batch, Channels, H, W)
        B, N, C = patch_tokens.shape
        # Calculate H and W dynamically based on input aspect ratio
        # (Assuming input H, W were valid multiples of 14)
        H_patches = int(x.shape[2] / 14)
        W_patches = int(x.shape[3] / 14)
        
        # Reshape to spatial grid
        spatial_features = patch_tokens.permute(0, 2, 1).reshape(B, C, H_patches, W_patches)
        
        # Apply GeM and Head
        pooled = self.pooling(spatial_features).flatten(1)
        return self.head(pooled)
    


class GeoCLIP(nn.Module):
    def __init__(self, dropout_rate=0.4, model_name="openai/clip-vit-base-patch16"):
        super(GeoCLIP, self).__init__()
        
        print(f"Loading {model_name}...")
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get embedding dimension (768 for base, 512 for patch32)
        embed_dim = self.backbone.config.hidden_size
        
        # Regression head - keep it simple
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 2)  # lat, lon
        )

    def forward(self, x):
        # Get pooled output (CLS token representation)
        outputs = self.backbone(pixel_values=x)
        pooled = outputs.pooler_output  # (Batch, embed_dim)
        
        return self.head(pooled)
    
    def unfreeze_last_layers(self, num_layers=2):
        """Optional: unfreeze last transformer blocks for fine-tuning"""
        print(f"Unfreezing last {num_layers} transformer layers...")
        for layer in self.backbone.vision_model.encoder.layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
