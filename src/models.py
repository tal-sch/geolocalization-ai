import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPVisionModel, ConvNextModel
from transformers import AutoModel, AutoConfig


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
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.model_name = "resnet50"
        
        # 1. Load Standard ResNet50
        # We use the default weights (IMAGENET1K_V2 is the modern standard)
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.backbone = models.resnet50(weights=weights)
        
        # 2. Freeze Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Replace the Head
        # ResNet's final layer is called 'fc' and has 2048 input features
        num_features = self.backbone.fc.in_features
        
        # Replace 'fc' with our regression head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # Lat/Lon
        )

    def forward(self, x):
        # ResNet is simple: just pass x through
        return self.backbone(x)

    def unfreeze_last_layers(self, num_layers=2):
        print(f"🔓 Unfreezing last {num_layers} blocks for ResNet50...")
        
        # ResNet structure: layer1, layer2, layer3, layer4
        # We unfreeze from the end backwards
        
        layers_to_unfreeze = []
        
        if num_layers >= 1: layers_to_unfreeze.append(self.backbone.layer4)
        if num_layers >= 2: layers_to_unfreeze.append(self.backbone.layer3)
        if num_layers >= 3: layers_to_unfreeze.append(self.backbone.layer2)
        if num_layers >= 4: layers_to_unfreeze.append(self.backbone.layer1)
            
        for block in layers_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
                
        # Always ensure head (fc) is unfrozen
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

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



class GeoDINO(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", dropout_rate=0.5):
        super().__init__()
        print(f"Loading DINOv2 Backbone: {model_name}...")
        
        # 1. Load DINOv2 (AutoModel handles the architecture)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Get hidden size (768 for Base, 384 for Small, 1024 for Large)
        self.embed_dim = self.backbone.config.hidden_size
        
        # 2. Regression Head
        # SWITCHED to LayerNorm to match CLIP (Better for Transformer embeddings)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 2) # Lat/Lon
        )
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # DINOv2 outputs a sequence of features
        outputs = self.backbone(x)
        
        # We take the [CLS] token (Index 0) which represents the whole image
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        return self.head(cls_token)

    def unfreeze_last_layers(self, num_layers=2):
        print(f"🔓 Unfreezing last {num_layers} layers for DINOv2...")
        # DINOv2 uses a standard transformer encoder structure
        encoder_layers = self.backbone.encoder.layer
        
        if num_layers > 0:
            for layer in encoder_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Always unfreeze head
        for param in self.head.parameters():
            param.requires_grad = True
    


class GeoCLIP(nn.Module):
    def __init__(self, dropout_rate=0.4, model_name="openai/clip-vit-base-patch32"):
        super(GeoCLIP, self).__init__()
        
        print(f"Loading {model_name}...")
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get embedding dimension (768 for base, 512 for patch32)
        embed_dim = self.backbone.config.hidden_size
        
        # Regression head
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
        print(f"🔓 Unfreezing last {num_layers} layers for CLIP...")
        
        # CLIPVisionModel wraps the actual model in .vision_model
        # Path: self.backbone.vision_model.encoder.layers
        layers = self.backbone.vision_model.encoder.layers
        
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        # Always unfreeze head
        for param in self.head.parameters():
            param.requires_grad = True


    
class GeoConvNeXt(nn.Module):
    def __init__(self, model_name="facebook/convnext-base-224", dropout_rate=0.5):
        super().__init__()
        print(f"Loading ConvNeXt: {model_name}...")
        
        # Load ConvNeXt backbone
        self.backbone = ConvNextModel.from_pretrained(model_name)
        
        # Get the hidden size from config
        # ConvNeXt outputs (batch, hidden_size, H, W)
        # We need to pool spatially and get a single vector
        self.embed_dim = self.backbone.config.hidden_sizes[-1]  # Last layer channels
        
        # Global Average Pooling (will be applied in forward)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 2)  # Lat/Lon
        )
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Get ConvNeXt features
        outputs = self.backbone(x)
        
        # outputs.last_hidden_state shape: (batch, hidden_size, H, W)
        features = outputs.last_hidden_state
        
        # Global average pooling
        pooled = self.pool(features)  # (batch, hidden_size, 1, 1)
        pooled = pooled.flatten(1)     # (batch, hidden_size)
        
        return self.head(pooled)
    
    def unfreeze_last_layers(self, num_layers=2):
        """
        ConvNeXt has 4 stages. Unfreeze from the end backwards.
        num_layers=2 means unfreeze stages 3 and 4
        num_layers=4 means unfreeze all stages
        """
        print(f"🔓 Unfreezing last {num_layers} stages for ConvNeXt...")
        
        # ConvNeXt structure: encoder.stages[0], [1], [2], [3]
        stages = self.backbone.encoder.stages
        
        # Unfreeze from the end
        stages_to_unfreeze = stages[-num_layers:] if num_layers < len(stages) else stages
        
        for stage in stages_to_unfreeze:
            for param in stage.parameters():
                param.requires_grad = True
        
        # Always unfreeze head
        for param in self.head.parameters():
            param.requires_grad = True


            