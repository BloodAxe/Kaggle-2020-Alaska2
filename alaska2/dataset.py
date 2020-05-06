import os
import random
from typing import Tuple, Optional, Union, List

import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

from .augmentations import get_augmentations

INPUT_IMAGE_KEY = "input_image"
INPUT_DCT_KEY = "input_dct"
INPUT_ELA_KEY = "input_ela"
INPUT_IMAGE_ID_KEY = "image_id"
INPUT_TRUE_MODIFICATION_TYPE = "true_modification_true"
INPUT_TRUE_MODIFICATION_FLAG = "true_modification_flag"

OUTPUT_PRED_MODIFICATION_FLAG = "pred_modification_flag"
OUTPUT_PRED_MODIFICATION_TYPE = "pred_modification_type"

__all__ = [
    "TrainingValidationDataset",
    "get_datasets",
    "compute_dct",
    "compute_ela",
    "get_test_dataset",
    "INPUT_TRUE_MODIFICATION_TYPE",
    "INPUT_TRUE_MODIFICATION_FLAG",
    "INPUT_DCT_KEY",
    "INPUT_ELA_KEY",
    "INPUT_IMAGE_KEY",
    "INPUT_IMAGE_ID_KEY",
    "OUTPUT_PRED_MODIFICATION_FLAG",
    "OUTPUT_PRED_MODIFICATION_TYPE",
]


def compute_dct(image):
    assert image.shape[0] % 8 == 0
    assert image.shape[1] % 8 == 0

    dct_image = np.zeros((image.shape[0] // 8, image.shape[1] // 8, 64), dtype=np.float32)

    one_over_255 = np.float32(1.0 / 255.0)
    image = image * one_over_255
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            dct = cv2.dct(image[i: i + 8, j: j + 8])
            dct_image[i // 8, j // 8, :] = dct.flatten()

    return dct_image


def compute_ela(image, quality_steps=[75, 80, 85, 90, 95]):
    diff = np.zeros((image.shape[0], image.shape[1], 3 * len(quality_steps)), dtype=np.float32)

    for i, q in enumerate(quality_steps):
        retval, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
        image_lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        np.subtract(image_lq, image, out=diff[..., i * 3: i * 3 + 3], dtype=np.float32)

    return diff


class TrainingValidationDataset(Dataset):
    def __init__(
            self, images: np.ndarray, targets: Optional[Union[List, np.ndarray]], transform: A.Compose, need_dct=False,
            need_ela=False
    ):
        self.images = images
        self.targets = targets
        self.transform = transform
        self.need_dct = need_dct
        self.need_ela = need_ela

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = self.transform(image=image)["image"]

        sample = {
            INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.images[index]),
            INPUT_IMAGE_KEY: tensor_from_rgb_image(image),
        }

        if self.targets is not None:
            sample[INPUT_TRUE_MODIFICATION_TYPE] = int(self.targets[index])
            sample[INPUT_TRUE_MODIFICATION_FLAG] = torch.tensor([self.targets[index] > 0]).float()

        if self.need_dct:
            sample[INPUT_DCT_KEY] = tensor_from_rgb_image(compute_dct(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)))

        if self.need_ela:
            sample[INPUT_ELA_KEY] = tensor_from_rgb_image(compute_ela(image))

        return sample


class BalancedTrainingDataset(Dataset):
    def __init__(self, images: np.ndarray, transform: A.Compose, need_dct=False, need_ela=False):
        self.images = images
        self.transform = transform
        self.need_dct = need_dct
        self.need_ela = need_ela
        self.methods = ["JMiPOD", "JUNIWARD", "UERD"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # With 50% probability select one of 3 altered images
        if random.random() > 0.5:
            target = random.randint(0, len(self.methods) - 1)
            method = self.methods[target]
            image = cv2.imread(self.images[index].replace("Cover", method))
            target = target + 1
        else:
            image = cv2.imread(self.images[index])
            target = 0

        image = self.transform(image=image)["image"]

        sample = {
            INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.images[index]),
            INPUT_IMAGE_KEY: tensor_from_rgb_image(image),
            INPUT_TRUE_MODIFICATION_TYPE: int(target),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([target > 0]).float()
        }

        if self.need_dct:
            sample[INPUT_DCT_KEY] = tensor_from_rgb_image(compute_dct(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)))
        if self.need_ela:
            sample[INPUT_ELA_KEY] = tensor_from_rgb_image(compute_ela(image))

        return sample


def get_datasets(
        data_dir: str,
        fold: int,
        image_size: Tuple[int, int],
        augmentation: str,
        fast: bool,
        need_dct=False,
        need_ela=False,
):
    train_transform = get_augmentations(augmentation)
    valid_transform = A.NoOp()

    if fold is None:
        if fast:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

            class_0 = fs.find_images_in_dir(os.path.join(data_dir, "Cover"))
            class_1 = fs.find_images_in_dir(os.path.join(data_dir, "JMiPOD"))
            class_2 = fs.find_images_in_dir(os.path.join(data_dir, "JUNIWARD"))
            class_3 = fs.find_images_in_dir(os.path.join(data_dir, "UERD"))

            train_x = valid_x = np.array(class_0 + class_1 + class_2 + class_3)
            train_y = valid_y = np.array(
                [0] * len(class_0) + [1] * len(class_1) + [2] * len(class_2) + [3] * len(class_3)
            )

            sampler = WeightedRandomSampler(np.ones(len(train_x)), 512)

            train_ds = TrainingValidationDataset(
                train_x, train_y, transform=train_transform, need_dct=need_dct, need_ela=need_ela
            )
            valid_ds = TrainingValidationDataset(
                valid_x, valid_y, transform=valid_transform, need_dct=need_dct, need_ela=need_ela
            )
            return train_ds, valid_ds, sampler
        else:
            train_class_0, test_class_0 = train_test_split(
                fs.find_images_in_dir(os.path.join(data_dir, "Cover")), test_size=1250, shuffle=True, random_state=42
            )
            train_class_1, test_class_1 = train_test_split(
                fs.find_images_in_dir(os.path.join(data_dir, "JMiPOD")), test_size=1250, shuffle=True, random_state=42
            )
            train_class_2, test_class_2 = train_test_split(
                fs.find_images_in_dir(os.path.join(data_dir, "JUNIWARD")),
                test_size=1250,
                shuffle=True,
                random_state=42,
            )
            train_class_3, test_class_3 = train_test_split(
                fs.find_images_in_dir(os.path.join(data_dir, "UERD")), test_size=1250, shuffle=True, random_state=42
            )

            train_x = np.array(train_class_0 + train_class_1 + train_class_2 + train_class_3)
            train_y = np.array(
                [0] * len(train_class_0)
                + [1] * len(train_class_1)
                + [2] * len(train_class_2)
                + [3] * len(train_class_3)
            )

            valid_x = np.array(test_class_0 + test_class_1 + test_class_2 + test_class_3)
            valid_y = np.array(
                [0] * len(test_class_0) + [1] * len(test_class_1) + [2] * len(test_class_2) + [3] * len(test_class_3)
            )
            sampler = None

            train_ds = TrainingValidationDataset(
                train_x, train_y, transform=train_transform, need_dct=need_dct, need_ela=need_ela
            )
            valid_ds = TrainingValidationDataset(
                valid_x, valid_y, transform=valid_transform, need_dct=need_dct, need_ela=need_ela
            )
            return train_ds, valid_ds, sampler
    else:
        if fast:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

        original_images = np.array(fs.find_images_in_dir(os.path.join(data_dir, "Cover")))
        image_sizes = np.array([os.stat(fname).st_size for fname in original_images])
        order = np.argsort(image_sizes)
        original_images = original_images[order]
        num_folds = 4
        num_images = len(original_images)

        folds_lut = (list(range(num_folds)) * num_images)[:num_images]
        folds_lut = np.array(folds_lut)

        train_images = original_images[folds_lut != fold].tolist()
        train_x = train_images.copy()
        train_y = [0] * len(train_images)

        valid_images = original_images[folds_lut == fold].tolist()
        valid_x = valid_images.copy()
        valid_y = [0] * len(valid_images)

        for i, method in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
            train_x += [fname.replace("Cover", method) for fname in train_images]
            train_y += [i + 1] * len(valid_images)

            valid_x += [fname.replace("Cover", method) for fname in valid_images]
            valid_y += [i + 1] * len(valid_images)

        train_ds = TrainingValidationDataset(
            train_x, train_y, transform=valid_transform, need_dct=need_dct, need_ela=need_ela
        )
        valid_ds = TrainingValidationDataset(
            valid_x, valid_y, transform=valid_transform, need_dct=need_dct, need_ela=need_ela
        )
        sampler = None
        return train_ds, valid_ds, sampler


def get_test_dataset(data_dir, need_dct):
    valid_transform = A.NoOp()
    images = fs.find_images_in_dir(os.path.join(data_dir, "Test"))
    return TrainingValidationDataset(images, None, valid_transform, need_dct=need_dct)
