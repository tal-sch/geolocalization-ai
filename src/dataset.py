import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

class GeolocalizationDataset(Dataset):
  def __init__(self, image_paths, coordinates, use_dropout=False, use_dropout2d=False, is_train =False):
    self.image_paths = image_paths
    self.coordinates = coordinates
    self.image_tensors = []

    print(f"Caching {len(image_paths)} images in RAM...")
    for path in tqdm(image_paths):
      with Image.open(path) as img:
        # Store as RGB Tensor in RAM to skip disk I/O later
        tensor_img = T.ToTensor()(img.convert('RGB'))
        self.image_tensors.append(tensor_img)

    transformations = []

    if is_train:
      self.transform = T.Compose([
          T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
          T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
          T.RandomGrayscale(p=0.1),
          T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711]
    )
            ])
      
    else:
        # Clean data for validation
        self.transform = T.Compose([
            T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711]
    )
        ])



  def __len__(self):
    return len(self.image_paths)


  def __getitem__(self, idx):
    image = self.image_tensors[idx]
    image = self.transform(image)

    coord = torch.tensor(self.coordinates[idx], dtype=torch.float32)

    return image, coord