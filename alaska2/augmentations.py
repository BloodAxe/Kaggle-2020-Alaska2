from typing import Tuple

import albumentations as A
import cv2
import random
import numpy as np
from albumentations.core.composition import BaseCompose


__all__ = ["RandomOrder", "EqualizeHistogram", "get_augmentations"]


class RandomOrder(BaseCompose):
    def __init__(self, transforms):
        super(RandomOrder, self).__init__(transforms, p=1)

    def __call__(self, force_apply=False, **data):
        transforms = list(self.transforms)
        random.shuffle(transforms)

        for idx, t in enumerate(transforms):
            data = t(force_apply=force_apply, **data)
        return data


class EqualizeHistogram(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, img, **params):
        equalized = cv2.equalizeHist(img)
        return cv2.addWeighted(equalized, 0.5, img, 0.5, 0, dtype=cv2.CV_8U)


def get_augmentations(augmentations_level: str):
    augmentations_level = augmentations_level.lower()
    if augmentations_level == "none":
        return A.NoOp()

    if augmentations_level == "safe":
        return A.Compose([A.HorizontalFlip(), A.VerticalFlip(),])

    if augmentations_level == "light":
        return RandomOrder([A.RandomRotate90(), A.Transpose(),])

    if augmentations_level == "medium":
        return RandomOrder([A.RandomRotate90(), A.Transpose(), A.RandomBrightnessContrast(p=0.3),])

    if augmentations_level == "hard":
        return A.Compose(
            [A.RandomRotate90(), A.Transpose(), A.RandomBrightnessContrast(p=0.3), A.RandomGridShuffle(grid=(8, 8))]
        )

    raise KeyError(augmentations_level)
