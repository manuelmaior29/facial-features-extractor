import collections
import torchvision
import torch
import torchvision.transforms.functional as F
import random
import numbers
import numpy as np
from PIL import Image

class ExtCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ipt, tgt):
        for t in self.transforms:
            ipt, tgt = t(ipt, tgt)
        return ipt, tgt

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ExtToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
        
    def __call__(self, ipt_image, tgt_image):
        if self.normalize:
            return F.to_tensor(ipt_image), torch.from_numpy(np.array(tgt_image, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(ipt_image, dtype=np.uint8)), torch.from_numpy(np.array(tgt_image, dtype=self.target_type))
        
    def __repr__(self):
        return self.__class__.__name__ + f'(normalize={self.normalize}, target_type={self.target_type})'

class ExtRandomHorizontalFlip(object):
    def __init__(self, p) -> None:
        self.p = p
        
    def __call__(self, ipt, tgt):
        if random.random() < self.p:
            return F.hflip(ipt), F.hflip(tgt)
        return ipt, tgt
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'
    
class ExtRandomVerticalFlip(object):
    def __init__(self, p) -> None:
        self.p = p
        
    def __call__(self, ipt, tgt):
        if random.random() < self.p:
            return F.vflip(ipt), F.vflip(tgt)
        return ipt, tgt
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'
    
class ExtNormalize(object):
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std
        
    def __call__(self, ipt, tgt):
        return F.normalize(tensor=ipt, mean=self.mean, std=self.std), tgt
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    
class ExtRandomCrop(object):
    def __init__(self, crop_size) -> None:
        self.crop_size = crop_size

    def __call__(self, ipt, tgt):
        if ipt.shape[1] < self.crop_size or ipt.shape[2] < self.crop_size:
            return ipt, tgt
        pos_x = random.randint(0, ipt.shape[1] - self.crop_size)
        pos_y = random.randint(0, ipt.shape[2] - self.crop_size)
        return F.crop(ipt, pos_x, pos_y, self.crop_size, self.crop_size), F.crop(tgt, pos_x, pos_y, self.crop_size, self.crop_size)
        
        