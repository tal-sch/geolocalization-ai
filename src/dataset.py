import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

class GeolocalizationDataset(Dataset):
  def __init__(self, image_paths, coordinates, target_size = (192, 256), is_train =False):
    self.image_paths = image_paths
    self.coordinates = coordinates
    self.image_tensors = []

    self.target_size = target_size

    print(f"Caching {len(image_paths)} images in RAM...")
    for path in tqdm(image_paths):
      with Image.open(path) as img:
        # Store as RGB Tensor in RAM to skip disk I/O later
        img = img.resize((self.target_size[0], self.target_size[1])).convert('RGB')
        tensor_img = T.ToTensor()(img)
        self.image_tensors.append(tensor_img)


  def __len__(self):
    return len(self.image_paths)


  def __getitem__(self, idx):
    image = self.image_tensors[idx]

    coord = torch.tensor(self.coordinates[idx], dtype=torch.float32)

    return image, coord