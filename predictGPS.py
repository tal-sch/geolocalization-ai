# --- Standard Library Imports ---
import os
import joblib
from PIL import Image

# --- Third-Party Imports ---
import numpy as np
import torchvision.transforms as T
import torch

# --- Local Application Imports ---
from src.models import MultiTaskDINOGeo

torch.backends.cudnn.benchmark = True

MODEL_PATH = "trained_models/dino_geo_model_best_mean_12.4.pth"
SCALER_PATH = "coordinate_scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_IMAGE_SIZE = (182, 252)  # (width, height)

model = None
scaler = None

def init_model_and_scaler():
    """Initialize and load the pre-trained model and scaler."""
    global model, scaler

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        print(f"Loading model on {DEVICE}...")
        model = MultiTaskDINOGeo(num_zones=25)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        model.eval().to(DEVICE)

    if scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)


def predict_gps(image: np.ndarray) -> np.ndarray:
    """Predict GPS coordinates from an input image.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        shape: (H, W, 3) in RGB format.
        Dtype: np.uint8
        Value range: [0, 255]

    Returns:
        np.ndarray: Predicted GPS coordinates as a NumPy array of shape (2,).
    """
    if model is None or scaler is None:
        init_model_and_scaler()

    image = Image.fromarray(image)
    image = image.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)

    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        normalized_pred_coords, _ = model(img_tensor)

    pred_coords = scaler.inverse_transform(normalized_pred_coords.cpu().numpy())
    return pred_coords.squeeze().astype(np.float32)

if __name__ == "__main__":
    IMAGE_FOLDER = "data_processed_manual_gps_252x182"

    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(reverse=True)  # Process most recent files first

    results = []

    # 2. Loop through images
    for i, img_path in enumerate(image_files[:100]):
        try:
            # Load Image (Simulating the grading environment)
            # The spec requires: np.ndarray, RGB, uint8
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_np = np.array(img)
            
            # --- CALL YOUR PIPELINE ---
            pred_lat, pred_lon = predict_gps(img_np)
            
            print(f"✅ Processed {os.path.basename(img_path)}: Predicted GPS ({pred_lat:.6f}, {pred_lon:.6f})")
            
        except Exception as e:
            print(f"⚠️ Error processing {os.path.basename(img_path)}: {e}")