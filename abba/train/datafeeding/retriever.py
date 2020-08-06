import jpegio as jio
import pandas as pd
import numpy as np
import pickle
import cv2
import albumentations as A
import os
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch
import sys
sys.path.insert(1,'./')
from train.tools.jpeg_utils import *

DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')

def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class TrainRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, decoder='NR', transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.decoder = decoder

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        
        if  self.decoder == 'NR':
            tmp = jio.read(f'{DATA_ROOT_PATH}/{kind}/{image_name}')
            image = decompress_structure(tmp)
            image = ycbcr2rgb(image).astype(np.float32)
            image /= 255.0
        else:
            image = cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        target = onehot(4, label)
        return image, target, image_name

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    
    
class TestRetriever(Dataset):

    def __init__(self, image_names, folder, decoder='NR', transforms=get_valid_transforms(), func_transforms=lambda x: x):
        super().__init__()
        self.image_names = np.array(image_names)
        self.transforms = transforms
        self.func_transforms = func_transforms
        self.folder = folder
        self.decoder = decoder

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        if  self.decoder == 'NR':
            tmp = jio.read(f'{self.folder}/{image_name}')
            image = decompress_structure(tmp).astype(np.float32)
            image = ycbcr2rgb(image)
            image /= 255.0
        else:
            image = cv2.imread(f'{self.folder}/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            
        image = self.func_transforms(image)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]
    