import numpy as np
from albumentations import RandomCrop, Compose, PadIfNeeded
if __name__ == '__main__':
    data = np.zeros((256, 1, 512))
    padding = Compose([PadIfNeeded(512, 1, p=1)], p=1)
    data = padding(image=data)
    data = data['image']
    print(data.shape)
    data = np.zeros((256, 1, 513))
    padding = Compose([PadIfNeeded(512, 1, p=1)], p=1)
    data = padding(image=data)
    data = data['image']
    print(data.shape)