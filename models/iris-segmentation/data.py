import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class IrisDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
class CASIAIris(IrisDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.ipts = os.listdir(f'{self.root}\\{self.sub_dir}\\input')
        self.tgts = os.listdir(f'{self.root}\\{self.sub_dir}\\segmentation')
        
    def __len__(self):
        return len(self.ipts)
    
    def __getitem__(self, index):
        try:
            ipt = Image.open(f'{self.ipts[index]}')
            tgt = Image.open(f'{self.tgts[index]}')
            if self.transform:
                ipt, tgt = self.transform(ipt, tgt)
            return ipt, tgt
        except:
            print(f'Could not get data pair at index {index}!')
    