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
    dct2channels_last,
    dct2spatial,
)

__all__ = [
    "get_augmentations",
    "get_obliterate_augs",
    "DctTranspose",
    "DctRandomRotate90",
    "dct_transpose",
    "dct_transpose_fast",
    "dct_rot90",
    "dct_rot90_fast",
    "RandomCrop8"
]


def change_even_rows_sign(dct_block):
    mask = np.ones_like(dct_block)
    mask[1::2, :] = -1
    return dct_block * mask


def change_even_cols_sign(dct_block):
    mask = np.ones_like(dct_block)
    mask[:, 1::2] = -1
    return dct_block * mask


def dct_transpose_block(dct_block: np.ndarray) -> np.ndarray:
    result = dct_block.transpose(1, 0)
    return result


def get_rot90_block(k: int, dtype):
    assert 0 <= k < 4
    mask = np.ones((8, 8), dtype=dtype)
    if k == 1:
        mask[:, 1::2] = -1
    elif k == 2:
        mask = get_rot90_block(1, dtype) * get_rot90_block(3, dtype)
    elif k == 3:
        mask[1::2, :] = -1

    return mask


def dct_rot90_block(dct_block: np.ndarray, k: int) -> np.ndarray:
    assert 0 <= k < 4
    if k == 1:
        dct_block = change_even_cols_sign(dct_block)
        dct_block = np.ascontiguousarray(np.transpose(dct_block))

    if k == 2:
        dct_block = change_even_cols_sign(dct_block)
        dct_block = change_even_rows_sign(dct_block)

    if k == 3:
        dct_block = change_even_rows_sign(dct_block)
        dct_block = np.ascontiguousarray(np.transpose(dct_block))

    return dct_block


def dct_rot90(dct_image: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return dct_image

    # Extract image planes
    y, cb, cr = dct_image[..., 0], dct_image[..., 1], dct_image[..., 2]

    # Reshape & permute to get [H//8, W//8, 64]
    y = dct2channels_last(y)
    cb = dct2channels_last(cb)
    cr = dct2channels_last(cr)

    # Now do spatial rotation of all 8x8 blocks
    y = np.rot90(y, k)
    cb = np.rot90(cb, k)
    cr = np.rot90(cr, k)

    # Now rotate each dct block individually
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = dct_rot90_block(y[i, j].reshape((8, 8)), k).flatten()
            cb[i, j] = dct_rot90_block(cb[i, j].reshape((8, 8)), k).flatten()
            cr[i, j] = dct_rot90_block(cr[i, j].reshape((8, 8)), k).flatten()

    # Now reshape & permute back to 512x512 size
    y = dct2spatial(y)
    cb = dct2spatial(cb)
    cr = dct2spatial(cr)
    return np.dstack([y, cb, cr])


def dct_rot90_fast(dct_image: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return dct_image

    # Extract image planes
    rows, cols, ch = dct_image.shape

    dct_image = dct_image.reshape((rows // 8, 8, cols // 8, 8, ch))

    # Now do spatial rotation
    dct_image = np.rot90(dct_image, k, axes=(0, 2))

    block = get_rot90_block(k, dtype=dct_image.dtype)
    block = block.reshape((1, 8, 1, 8, 1))

    dct_image = dct_image * block

    if k == 1 or k == 3:
        dct_image = np.transpose(dct_image, axes=(0, 3, 2, 1, 4))

    dct_image = dct_image.reshape((rows, cols, ch))
    return np.ascontiguousarray(dct_image)


def dct_transpose(dct_image: np.ndarray) -> np.ndarray:
    # Extract image planes
    y, cb, cr = dct_image[..., 0], dct_image[..., 1], dct_image[..., 2]

    # Reshape & permute to get [H//8, W//8, 64]
    y = dct2channels_last(y)
    cb = dct2channels_last(cb)
    cr = dct2channels_last(cr)

    # Now do spatial transpose of all 8x8 blocks
    y = y.transpose(1, 0, 2)
    cb = cb.transpose(1, 0, 2)
    cr = cr.transpose(1, 0, 2)

    # Now rotate each dct block individually
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = dct_transpose_block(y[i, j].reshape((8, 8))).flatten()
            cb[i, j] = dct_transpose_block(cb[i, j].reshape((8, 8))).flatten()
            cr[i, j] = dct_transpose_block(cr[i, j].reshape((8, 8))).flatten()

    # Now reshape & permute back to 512x512 size
    y = dct2spatial(y)
    cb = dct2spatial(cb)
    cr = dct2spatial(cr)
    return np.dstack([y, cb, cr])


def dct_transpose_fast(dct_image: np.ndarray) -> np.ndarray:
    # Extract image planes
    rows, cols, ch = dct_image.shape

    dct_image = dct_image.reshape((rows // 8, 8, cols // 8, 8, ch))

    # Now do spatial rotation
    dct_image = np.transpose(dct_image, axes=(2, 1, 0, 3, 4))
    dct_image = np.transpose(dct_image, axes=(0, 3, 2, 1, 4))

    dct_image = dct_image.reshape((rows, cols, ch))
    return np.ascontiguousarray(dct_image)


class DctRandomRotate90(A.RandomRotate90):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "input_dct": self.apply_dct,
        }

    def apply_dct(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return dct_rot90_fast(img, factor)


class DctTranspose(A.Transpose):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply_dct(self, img, **params):
        return dct_transpose_fast(img)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "input_dct": self.apply_dct,
        }


class RandomCrop8(A.RandomCrop):
    def apply(self, img, h_start=0, w_start=0, **params):
        height, width = img.shape[:2]
        if height < self.height or width < self.width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=self.height, crop_width=self.width, height=height, width=width
                )
            )
        from albumentations.augmentations.functional import get_random_crop_coords

        x1, y1, x2, y2 = get_random_crop_coords(height, width, self.height, self.width, h_start, w_start)
        x1 = (x1 // 8) * 8
        y1 = (y1 // 8) * 8
        x2 = x1 + self.width
        y2 = y1 + self.height
        img = img[y1:y2, x1:x2]
        return img


def get_obliterate_augs():
    """
    Get the augmentation that can obliterate the hidden signal.
    This is used as augmentation to create negative sample from positive one.
    :return:
    """
    return A.OneOf(
        [
            A.ImageCompression(quality_lower=70, quality_upper=95, p=1),
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
                DctRandomRotate90(p=1.0),
                DctTranspose(p=0.5),
            ],
            additional_targets=additional_targets,
        )

    if augmentations_level == "medium":
        return A.ReplayCompose(
            [
                maybe_crop,
                DctRandomRotate90(p=1.0),
                DctTranspose(p=0.5),
                A.CoarseDropout(max_holes=1, min_height=32, max_height=256, min_width=32, max_width=256, p=0.2),
            ],
            additional_targets=additional_targets,
        )

    if augmentations_level == "hard":
        return A.ReplayCompose(
            [
                maybe_crop,
                DctRandomRotate90(p=1.0),
                DctTranspose(p=0.5),
                A.OneOf(
                    [
                        A.RandomGridShuffle(grid=(2, 2)),
                        A.RandomGridShuffle(grid=(3, 3)),
                        A.RandomGridShuffle(grid=(4, 4)),
                    ],
                    p=0.1,
                ),
                A.CoarseDropout(max_holes=1, min_height=32, max_height=256, min_width=32, max_width=256, p=0.2),
            ],
            additional_targets=additional_targets,
        )

    raise KeyError(augmentations_level)
