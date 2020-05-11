from typing import Tuple

import albumentations as A
import cv2
import random
import numpy as np
from albumentations.core.composition import BaseCompose
from .dataset import (
    INPUT_FEATURES_DCT_CR_KEY,
    INPUT_FEATURES_DCT_Y_KEY,
    INPUT_FEATURES_DCT_CB_KEY,
    INPUT_FEATURES_BLUR_KEY,
    INPUT_FEATURES_ELA_KEY,
)

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


def get_augmentations(augmentations_level: str, image_size: Tuple[int, int]):
    if image_size[0] != 512 or image_size[1] != 512:
        print("Adding RandomCrop size target image size is", image_size)
        maybe_crop = A.RandomCrop(image_size[0], image_size[1], always_apply=True)
    else:
        maybe_crop = A.NoOp()

    additional_targets = {
        INPUT_FEATURES_ELA_KEY: "image",
        INPUT_FEATURES_BLUR_KEY: "image",
        INPUT_FEATURES_DCT_Y_KEY: "image",
        INPUT_FEATURES_DCT_CB_KEY: "image",
        INPUT_FEATURES_DCT_CR_KEY: "image",
    }
    augmentations_level = augmentations_level.lower()
    if augmentations_level == "none":
        return A.NoOp()

    if augmentations_level == "safe":
        return A.ReplayCompose(
            [maybe_crop, A.HorizontalFlip(), A.VerticalFlip()], additional_targets=additional_targets
        )

    if augmentations_level == "light":
        return A.ReplayCompose([maybe_crop, A.RandomRotate90(), A.Transpose()], additional_targets=additional_targets)

    if augmentations_level == "medium":
        return A.ReplayCompose(
            [
                maybe_crop,
                A.RandomRotate90(),
                A.Transpose(),
                A.OneOf(
                    [
                        A.RandomGridShuffle(grid=(2, 2)),
                        A.RandomGridShuffle(grid=(3, 3)),
                        A.RandomGridShuffle(grid=(4, 4)),
                    ]
                ),
                A.CoarseDropout(min_width=8, min_height=8, max_width=256, max_height=256, max_holes=3),
            ],
            additional_targets=additional_targets,
        )

    if augmentations_level == "hard":
        return A.ReplayCompose(
            [
                maybe_crop,
                A.RandomRotate90(),
                A.Transpose(),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGridShuffle(grid=(8, 8)),
                A.ShiftScaleRotate(
                    rotate_limit=5, shift_limit=0.05, scale_limit=0.05, border_mode=cv2.BORDER_CONSTANT
                ),
            ],
            additional_targets=additional_targets,
        )

    raise KeyError(augmentations_level)
