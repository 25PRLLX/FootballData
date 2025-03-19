import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class GoalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the GoalDataset.

        Parameters:
        - root_dir (str): Root directory containing images and labels.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.labels = sorted(os.listdir(os.path.join(root_dir, 'labels')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.images[idx])
        label_path = os.path.join(self.root_dir, 'labels', self.labels[idx])
        
        image = Image.open(img_path).convert('RGB')
        with open(label_path, 'r') as f:
            label = int(f.read().strip())
        
        if self.transform:
            image = self.transform(image)
        
        return image, label