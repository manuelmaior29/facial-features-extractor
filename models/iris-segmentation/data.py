import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from utils.ext_transforms import ExtToTensor

class IrisDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
class CASIAIris(IrisDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.ipts = sorted(os.listdir(f'{self.root}/input'))
        self.tgts = sorted(os.listdir(f'{self.root}/segmentation'))
        
    def __len__(self):
        return len(self.ipts)
    
    def __getitem__(self, index):
        try:
            ipt = Image.open(f'{self.root}/input/{self.ipts[index]}')
            tgt = Image.open(f'{self.root}/segmentation/{self.tgts[index]}').convert('L')
            ipt, tgt = ExtToTensor()(ipt, tgt)
            tgt[tgt == 255] = 1
            if self.transform:
                ipt, tgt = self.transform(ipt, tgt)
                
            return ipt, tgt.long()
        except:
            print(f'Could not get data pair at index {index}!')
    