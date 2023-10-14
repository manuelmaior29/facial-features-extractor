import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class IrisDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.data_files)
        
    def __getitem__(self, idx):
        try:
            pass
        except:
            print(f'Could not get data item with index {idx}.')

