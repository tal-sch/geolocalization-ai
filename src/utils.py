from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import contextily as cx

# Function to extract GPS coordinates from image EXIF data
def extractCoordinates(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    if not exif_data:
        return None

    gps_info = {}
    for tag, value in exif_data.items():
        decoded = ExifTags.TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            # Map the inner GPS tags (e.g., 1: 'GPSLatitudeRef', 2: 'GPSLatitude'...)
            for t in value:
                sub_decoded = ExifTags.GPSTAGS.get(t, t)
                gps_info[sub_decoded] = value[t]

    # If GPS data exists, calculate the decimals
    if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
        lat_dms = gps_info["GPSLatitude"]
        lat_ref = gps_info["GPSLatitudeRef"]
        lon_dms = gps_info["GPSLongitude"]
        lon_ref = gps_info["GPSLongitudeRef"]

        # Helper to convert DMS to Decimal
        def dms_to_decimal(dms, ref):
            # dms is a list of 3 rationals [(deg_num, deg_den), (min, den), (sec, den)]
            deg = float(dms[0])
            minute = float(dms[1])
            sec = float(dms[2])

            decimal = deg + (minute / 60.0) + (sec / 3600.0)
            if ref in ["S", "W"]:
                decimal = -decimal
            return decimal

        lat = dms_to_decimal(lat_dms, lat_ref)
        lon = dms_to_decimal(lon_dms, lon_ref)

        return lat, lon

    return None


def aspect_crop(img):
    """
    Standardizes mixed datasets to 4:3 aspect ratio.
    - If image is 4:3, it does nothing
    - If image is 18:9, it crops Sky/Ground to match 4:3
    """
    w, h = img.size
    current_ratio = w / h
    target_ratio = 3 / 4  # 0.75

    # If already close to 4:3, do nothing
    if abs(current_ratio - target_ratio) < 0.05:
        return img

    # If image is too tall (Portrait), Crop top and bottom
    if current_ratio < target_ratio:
        # Calculate new height to achieve 4:3 based on current width
        new_h = int(w / target_ratio)
        pixels_to_remove = h - new_h

        # remove 30% from top (sky), then 70% from bottom (ground)
        top_crop = int(pixels_to_remove * 0.3)

        return TF.crop(img, top_crop, 0, new_h, w)

    # Fallback: If image is too wide (Landscape), CenterCrop it
    return TF.center_crop(img, (h * 3 // 4, h))


def haversine_distance(coord1, coord2):
    """
    Calculates the distance in meters between two points on the earth.
    Input: Arrays of [Latitude, Longitude]
    """
    # Earth's radius in meters
    R = 6371000.0

    # Convert degrees to radians
    lat1, lon1 = np.radians(coord1[:, 0]), np.radians(coord1[:, 1])
    lat2, lon2 = np.radians(coord2[:, 0]), np.radians(coord2[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# Function to plot 10 images and their filenames
def plot_images_from_dataloader(dataloader, num_images=10):
    # Get the first batch of images and labels
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Move images to CPU for visualization
    images = images[:num_images].cpu()

    # Plot the images
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        img = img * 0.229 + 0.485  # Unnormalize (mean and std from T.Normalize)
        img = img.clip(0, 1)  # Clip values to [0, 1] for display

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()

def setup_TensorBoard_writers():
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_log_dir = f"runs/experiment_{current_time}"

    # Create separate writers for training and validation
    writer_train = SummaryWriter(f"{base_log_dir}/train")
    writer_val   = SummaryWriter(f"{base_log_dir}/val")

    print(f"Tensorboard - Logging to: {base_log_dir}")

    return writer_train, writer_val

def log_error_map(preds, trues,  epoch, num_points = 50, TB_writer = None):
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # plot true coordinates (Green) and predicted coordinates (Red)
    ax.scatter(trues[:, 1], trues[:, 0], c='lime', label='True', alpha=0.8, s=30, edgecolors='black', zorder=2)
    ax.scatter(preds[:, 1], preds[:, 0], c='red', label='Pred', alpha=0.8, s=30, edgecolors='black', zorder=2)
    
    # draw lines between true and predicted points
    for i in range(min(num_points, len(trues))):
        ax.plot([trues[i, 1], preds[i, 1]], 
                [trues[i, 0], preds[i, 0]], 'gray', alpha=0.3)
    
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)

    try:
        cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik, alpha=1.0, reset_extent=False)
    except Exception as e:
        print(f"Could not fetch map tiles: {e}")

    ax.legend(loc='upper right')
    ax.set_title(f"Campus Geolocalization Analysis (Epoch {epoch})")
    
    if TB_writer is not None:
        # log to tensorboard
        TB_writer.add_figure("Analysis/Error_Map", fig, epoch)
        plt.close(fig) 
    else:
        plt.show()