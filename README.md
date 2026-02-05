---
layout: default
---
# 📍 Campus Geolocalization with DINOv2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning system for precise, single-image geolocalization within the Ben-Gurion University campus. This project leverages the self-supervised features of **DINOv2** (Vision Transformer) combined with a **Multi-Task Head** to predict both exact GPS coordinates and coarse-grained campus zones from a single ground-level photograph.

## 📖 Overview

Standard GPS often struggles in dense or semi-urban environments due to signal multipath and occlusion.  
**Visual Geolocalization** addresses this by inferring location directly from visual cues.

This project adopts a **Multi-Task Learning** approach to improve stability and robustness:

1. **Regression Task**  
   Predicts continuous, normalized latitude and longitude values.
2. **Classification Task**  
   Predicts a coarse campus *zone* (1 of 25 grid sectors).

### Key Features

- **Backbone**  
  Pre-trained **DINOv2 (ViT-Small/14)** for strong geometric and semantic representations.
- **Architecture**  
  Shared projection head branching into:
  - Regression Head (Huber Loss)
  - Classification Head (Cross-Entropy Loss)
- **Training Strategy**  
  Progressive unfreezing to adapt the foundation model to the campus domain without catastrophic forgetting.
- **Performance**  
  Mean localization error of **~12–13 meters** on the test set, significantly outperforming supervised baselines (ResNet50, ConvNeXt).

---

## 🛠️ Environment Setup

### Prerequisites

- Windows / Linux / macOS
- Python 3.8+
- CUDA-capable GPU *(recommended for training)*

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/tal-sch/geolocalization-ai.git
cd geolocalization-ai
```

#### 2. Create a Virtual Environment 
```bash
conda create -n geoloc python=3.9
conda activate geoloc
```
or
```bash
python -m venv .venv
.\.venv\Scripts\Activate
```
#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Ensure the CUDA enabled version of torch is installed   
Check https://pytorch.org/get-started/locally/ to find the correct version for your machine:
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Main dependencies:**  
`torch`, `torchvision`, `timm`, `pandas`, `numpy`, `Pillow`, `scikit-learn`, `tensorboard`

---

## 📂 Dataset Setup

The dataset contains **1,475 training images** and a validation set captured around the BGU campus.

### 1. Download Data

From the shared Google Drive:
- `dataset_root.zip`
- `dino_geo_model_best_mean_12.4.zip` (pre-trained weights)

🔗 **Google Drive:**  
https://drive.google.com/drive/folders/1buh4Y_Scxa9DXdy_Wwv0XUPVBQMsxPwN?usp=sharing

> **Note:** Access permissions may be required.

### 2. Extract and Organize

- Extract `dataset_root.zip` into the project root
- Create a `trained_models/` directory
- Place `dino_geo_model_best_mean_12.4.pth` inside it

Your directory structure should look like this:

```text
geolocalization-AI/
├── dataset_root/         # Extracted dataset
│   ├── images/
│   │   ├── IMG_001.jpg
│   │   └── ...
│   └── gt.csv            # Columns: filename, lat, lon
├── trained_models/
│   └── dino_geo_model_best_mean_12.4.pth
├── runs/                 # TensorBoard logs
│   └── (empty)
├── src/
│   └── ...
├── coordinate_scaler_main.pkl
├── prediction.ipynb
├── prediction.py
├── train.ipynb
└── train.py
```

---

## 🚀 Training

To train the multi-task model from scratch using **Progressive Unfreezing**, choose one of the following options:

### Option 1: Terminal
```bash
python train.py
```

### Option 2: Notebook
Open `train.ipynb` in Jupyter or VS Code and run the cells interactively.

- Hyperparameters (epochs, learning rate, batch size) can be modified at the top of both files.
- Model weights are saved automatically to `trained_models/` upon validation improvement.
- Training can be stopped at any time with **Ctrl+C**, or allowed to run until convergence.

### Monitoring with TensorBoard

Training metrics are logged automatically. To visualize loss curves and validation performance run:

```bash
tensorboard --logdir runs/
```

---

## 🎯 Prediction

A simple interface is provided for predicting GPS coordinates from a single image using the pre-trained model (**~12.5m error**).

### Example Usage

```python
from prediction import predict_gps
from PIL import Image
import numpy as np

# Path to input image
image_path = "dataset_root/images/IMG_123.jpg"

with Image.open(image_path) as img:
    img = img.convert("RGB")
    img_np = np.array(img)

# Predict GPS coordinates
pred_coords = predict_gps(img_np)

print(f"Predicted GPS ({pred_coords[0]:.6f}, {pred_coords[1]:.6f})")
# Example output: Predicted GPS (31.261230, 34.801550)
```

---

## 👥 Credits

- **Authors:** Tal S., Ilan R.  
- **Institution:** Ben-Gurion University of the Negev
