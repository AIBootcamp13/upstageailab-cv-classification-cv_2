import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DocumentDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.data.iloc[idx, 0]}"
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label