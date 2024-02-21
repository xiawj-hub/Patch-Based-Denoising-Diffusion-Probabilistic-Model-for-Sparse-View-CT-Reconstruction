import torch
import random
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms

def random_rot(img):
    k = np.random.randint(0, 3)
    img = np.rot90(img, k+1).copy()
    return img

def random_flip(img):
    axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    return img

def RadomGenerator(ndct):
    if random.random() > 0.5:
        ndct = random_rot(ndct)
    if random.random() > 0.5:
        ndct = random_flip(ndct)
    return ndct

class MayoDataset(Dataset):
    def __init__(self, root, split='train'):
        self.path = os.path.join(root, split)
        self.list = np.array(sorted(glob.glob(os.path.join(self.path, '*.np*'))))
        self.data_len = len(self.list)
        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize([256,256])
        self.split = split
        
    def __getitem__(self, index):
        data_path = self.list[index]
        data = np.load(data_path)
        path = data_path.split('/')[-1].split('.')[0]
        proj = data.astype(np.float32)
        if self.split == 'train':
            proj = RadomGenerator(proj)
        proj = self.transform(proj)
        return {'label':proj, 'path':path}

    def __len__(self):
        return self.data_len