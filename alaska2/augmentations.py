from typing import Tuple

import albumentations as A
import cv2
import random
import numpy as np
from albumentations.core.composition import BaseCompose

from .dataset import (
    INPUT_IMAGE_KEY,
    INPUT_FEATURES_ELA_KEY,
    INPUT_FEATURES_DCT_CR_KEY,
    INPUT_FEATURES_DCT_CB_KEY,
    INPUT_FEATURES_DCT_Y_KEY,
    INPUT_FEATURES_BLUR_KEY,
    INPUT_FEATURES_CHANNEL_CR_KEY,
    INPUT_FEATURES_CHANNEL_CB_KEY,
    INPUT_FEATURES_CHANNEL_Y_KEY,
)

__all__ = ["get_augmentations", "get_obliterate_augs"]


def get_obliterate_augs():
    """
    Get the augmentation that can obliterate the hidden signal.
    This is used as augmentation to create negative sample from positive one.
    :return:
    """
    return A.OneOf(
        [
            A.ImageCompression(quality_lower=70, quality_upper=90, p=1),
            A.RandomSizedCrop((256, 384), 512, 512, interpolation=cv2.INTER_CUBIC, p=1),
            A.RandomSizedCrop((256, 384), 512, 512, interpolation=cv2.INTER_LINEAR, p=1),
            A.Downscale(p=1),
            A.GaussianBlur(blur_limit=(5, 9), p=1),
        ],
        p=1,
    )


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
        INPUT_FEATURES_CHANNEL_Y_KEY: "image",
        INPUT_FEATURES_CHANNEL_CR_KEY: "image",
        INPUT_FEATURES_CHANNEL_CB_KEY: "image",
    }

    augmentations_level = augmentations_level.lower()
    if augmentations_level == "none":
        return A.ReplayCompose([A.NoOp()])

    if augmentations_level == "safe":
        return A.ReplayCompose(
            [maybe_crop, A.HorizontalFlip(), A.VerticalFlip()], additional_targets=additional_targets
        )

    if augmentations_level == "light":
        return A.ReplayCompose(
            [
                maybe_crop,
                # D4
                A.RandomRotate90(p=1.0),
                A.Transpose(p=0.5),
            ],
            additional_targets=additional_targets,
        )

    if augmentations_level == "medium":
        return A.ReplayCompose(
            [
                maybe_crop,
                A.RandomRotate90(p=1.0),
                A.Transpose(p=0.5),
                A.OneOf(
                    [
                        A.RandomGridShuffle(grid=(2, 2)),
                        A.RandomGridShuffle(grid=(3, 3)),
                        A.RandomGridShuffle(grid=(4, 4)),
                    ],
                    p=0.1,
                ),
                A.CoarseDropout(max_holes=1, min_height=128, max_height=256, min_width=128, max_width=256, p=0.1),
            ],
            additional_targets=additional_targets,
        )

    if augmentations_level == "hard":
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
                    ],
                    p=0.1,
                ),
                A.CoarseDropout(max_holes=1, min_height=128, max_height=256, min_width=128, max_width=256, p=0.1),
            ],
            additional_targets=additional_targets,
        )

    raise KeyError(augmentations_level)
