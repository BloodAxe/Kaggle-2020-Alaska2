import os
import random
from typing import Tuple, Optional, Union, List
import pandas as pd
import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset, WeightedRandomSampler

from .augmentations import get_augmentations

INPUT_IMAGE_KEY = "input_image"
INPUT_FEATURES_DCT_KEY = "input_dct"
INPUT_FEATURES_ELA_KEY = "input_ela"
INPUT_FEATURES_BLUR_KEY = "input_blur"
INPUT_IMAGE_ID_KEY = "image_id"
INPUT_FOLD_KEY = "fold"

INPUT_TRUE_MODIFICATION_TYPE = "true_modification_type"
INPUT_TRUE_MODIFICATION_FLAG = "true_modification_flag"

OUTPUT_PRED_MODIFICATION_TYPE = "pred_modification_type"
OUTPUT_PRED_MODIFICATION_FLAG = "pred_modification_flag"

OUTPUT_PRED_EMBEDDING = "pred_embedding"

OUTPUT_FEATURE_MAP_4 = "pred_fm_4"
OUTPUT_FEATURE_MAP_8 = "pred_fm_8"
OUTPUT_FEATURE_MAP_16 = "pred_fm_16"
OUTPUT_FEATURE_MAP_32 = "pred_fm_32"

__all__ = [
    "INPUT_FEATURES_BLUR_KEY",
    "INPUT_FEATURES_DCT_KEY",
    "INPUT_FEATURES_ELA_KEY",
    "INPUT_FOLD_KEY",
    "INPUT_IMAGE_ID_KEY",
    "INPUT_IMAGE_KEY",
    "INPUT_TRUE_MODIFICATION_FLAG",
    "INPUT_TRUE_MODIFICATION_TYPE",
    "OUTPUT_FEATURE_MAP_16",
    "OUTPUT_FEATURE_MAP_32",
    "OUTPUT_FEATURE_MAP_4",
    "OUTPUT_FEATURE_MAP_8",
    "OUTPUT_PRED_EMBEDDING",
    "OUTPUT_PRED_MODIFICATION_FLAG",
    "OUTPUT_PRED_MODIFICATION_TYPE",
    "TrainingValidationDataset",
    "compute_blur_features",
    "compute_dct",
    "compute_ela",
    "get_datasets",
    "get_datasets_batched",
    "get_test_dataset",
]


def compute_dct(jpeg_file):
    from jpeg2dct.numpy import load, loads

    dct_y, dct_cb, dct_cr = load("Cover/00001.jpg")

    return dct_y, dct_cb, dct_cr


DCTMTX = np.array(
    [
        [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
        [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
        [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
        [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
        [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
        [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
        [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
        [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975],
    ],
    dtype=np.float32,
)


def compute_ela(image, quality_steps=[75, 80, 85, 90, 95]):
    diff = np.zeros((image.shape[0], image.shape[1], 3 * len(quality_steps)), dtype=np.float32)

    for i, q in enumerate(quality_steps):
        retval, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
        image_lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        np.subtract(image_lq, image, out=diff[..., i * 3 : i * 3 + 3], dtype=np.float32)

    return diff


def compute_blur_features(image):
    image2 = cv2.pyrDown(image)
    image4 = cv2.pyrDown(image2)
    image8 = cv2.pyrDown(image4)
    image_size = image.shape[1], image.shape[0]

    image8 = cv2.resize(image8, image_size, interpolation=cv2.INTER_LINEAR)
    image4 = cv2.resize(image4, image_size, interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, image_size, interpolation=cv2.INTER_LINEAR)

    diff = np.zeros((image.shape[0], image.shape[1], 3 * 3), dtype=np.float32)

    np.subtract(image2, image, out=diff[..., 0:3], dtype=np.float32)
    np.subtract(image4, image, out=diff[..., 3:6], dtype=np.float32)
    np.subtract(image8, image, out=diff[..., 6:9], dtype=np.float32)
    return diff


def compute_features(image_fname, features):
    sample = {}

    if INPUT_FEATURES_DCT_KEY in features:
        dct_y, dct_cb, dct_cr = compute_dct(image_fname)
        dct_y = tensor_from_rgb_image(dct_y)
        dct_cb = tensor_from_rgb_image(dct_cb)
        dct_cr = tensor_from_rgb_image(dct_cr)
        dct = torch.cat([dct_y, dct_cb, dct_cr], dim=0)
        sample[INPUT_FEATURES_DCT_KEY] = dct

    if INPUT_IMAGE_KEY in features:
        image = cv2.imread(image_fname)
        sample[INPUT_IMAGE_KEY] = tensor_from_rgb_image(image)

    if INPUT_FEATURES_ELA_KEY in features:
        image = cv2.imread(image_fname)
        sample[INPUT_FEATURES_ELA_KEY] = tensor_from_rgb_image(compute_ela(image))

    if INPUT_FEATURES_BLUR_KEY in features:
        image = cv2.imread(image_fname)
        sample[INPUT_FEATURES_BLUR_KEY] = tensor_from_rgb_image(compute_blur_features(image))

    return sample


class TrainingValidationDataset(Dataset):
    def __init__(
        self, images: np.ndarray, targets: Optional[Union[List, np.ndarray]], transform: A.Compose, features: List[str]
    ):
        if targets is not None:
            if len(images) != len(targets):
                raise ValueError(f"Size of images and targets does not match: {len(images)} {len(targets)}")

        self.images = images
        self.targets = targets
        self.transform = transform
        self.features = features

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"TrainingValidationDataset(len={len(self)}, targets_hist={np.bincount(self.targets)}, features={self.features})"

    def __getitem__(self, index):
        image_fname = self.images[index]
        data = compute_features(image_fname, self.features)

        data = self.transform(**data)

        data[INPUT_IMAGE_ID_KEY] = fs.id_from_fname(self.images[index])

        if self.targets is not None:
            data[INPUT_TRUE_MODIFICATION_TYPE] = int(self.targets[index])
            data[INPUT_TRUE_MODIFICATION_FLAG] = torch.tensor([self.targets[index] > 0]).float()

        return data


class BalancedTrainingDataset(Dataset):
    def __init__(self, images: np.ndarray, transform: A.Compose, features):
        self.images = images
        self.transform = transform
        self.methods = ["JMiPOD", "JUNIWARD", "UERD"]
        self.features = features

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # With 50% probability select one of 3 altered images
        if random.random() >= 0.5:
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
            INPUT_TRUE_MODIFICATION_TYPE: int(target),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([target > 0]).float(),
        }

        sample.update(compute_features(image, self.features))
        return sample


class CoverImageDataset(Dataset):
    def __init__(self, images: np.ndarray, transform: A.Compose, features):
        self.images = images
        self.transform = transform
        self.features = features

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        target = 0

        image = self.transform(image=image)["image"]

        sample = {
            INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.images[index]),
            INPUT_TRUE_MODIFICATION_TYPE: int(target),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([target > 0]).float(),
        }

        sample.update(compute_features(image, self.features))
        return sample


class ModifiedImageDataset(Dataset):
    def __init__(self, images: np.ndarray, transform: A.Compose, features):
        self.images = images
        self.transform = transform
        self.methods = ["JMiPOD", "JUNIWARD", "UERD"]
        self.features = features

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select one of 3 altered images
        target = random.randint(0, len(self.methods) - 1)
        method = self.methods[target]
        image = cv2.imread(self.images[index].replace("Cover", method))
        target = target + 1

        image = self.transform(image=image)["image"]

        sample = {
            INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.images[index]),
            INPUT_TRUE_MODIFICATION_TYPE: int(target),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([target > 0]).float(),
        }

        sample.update(compute_features(image, self.features))
        return sample


class BatchedImageDataset(Dataset):
    def __init__(self, images: np.ndarray, transform: A.Compose, features):
        self.images = images
        self.features = features
        if isinstance(transform, A.ReplayCompose):
            self.transform = transform
        else:
            self.transform = A.ReplayCompose([transform])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select one of 3 altered images
        image0 = cv2.imread(self.images[index])
        image1 = cv2.imread(self.images[index].replace("Cover", "JMiPOD"))
        image2 = cv2.imread(self.images[index].replace("Cover", "JUNIWARD"))
        image3 = cv2.imread(self.images[index].replace("Cover", "UERD"))

        data = self.transform(image=image0)

        image0 = data["image"]
        image1 = self.transform.replay(data["replay"], image=image1)["image"]
        image2 = self.transform.replay(data["replay"], image=image2)["image"]
        image3 = self.transform.replay(data["replay"], image=image3)["image"]

        sample = {
            INPUT_IMAGE_ID_KEY: [fs.id_from_fname(self.images[index])] * 4,
            INPUT_IMAGE_KEY: torch.stack(
                [
                    tensor_from_rgb_image(image0),
                    tensor_from_rgb_image(image1),
                    tensor_from_rgb_image(image2),
                    tensor_from_rgb_image(image3),
                ]
            ),
            INPUT_TRUE_MODIFICATION_TYPE: torch.tensor([0, 1, 2, 3]).long(),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([0, 1, 1, 1]).float(),
        }
        # TODO
        # sample.update(compute_features(image, self.features))
        return sample


def get_datasets(
    data_dir: str,
    fold: int,
    augmentation: str = "light",
    fast: bool = False,
    image_size: Tuple[int, int] = (512, 512),
    balance=False,
    features=None,
):
    train_transform = get_augmentations(augmentation, image_size)
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

            train_ds = TrainingValidationDataset(train_x, train_y, transform=train_transform, features=features)
            valid_ds = TrainingValidationDataset(valid_x, valid_y, transform=valid_transform, features=features)

            print("Train", train_ds)
            print("Valid", valid_ds)

            return train_ds, valid_ds, sampler
        else:
            raise ValueError("Fold must be set")
    else:
        data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds.csv"))

        train_images = data_folds.loc[data_folds[INPUT_FOLD_KEY] != fold, INPUT_IMAGE_ID_KEY].tolist()
        valid_images = data_folds.loc[data_folds[INPUT_FOLD_KEY] == fold, INPUT_IMAGE_ID_KEY].tolist()

        train_images = [os.path.join(data_dir, "Cover", x) for x in train_images]
        valid_images = [os.path.join(data_dir, "Cover", x) for x in valid_images]

        train_x = train_images.copy()
        train_y = [0] * len(train_images)

        valid_x = valid_images.copy()
        valid_y = [0] * len(valid_images)

        for i, method in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
            train_x += [fname.replace("Cover", method) for fname in train_images]
            train_y += [i + 1] * len(train_images)

            valid_x += [fname.replace("Cover", method) for fname in valid_images]
            valid_y += [i + 1] * len(valid_images)

        train_ds = TrainingValidationDataset(train_x, train_y, transform=train_transform, features=features)
        valid_ds = TrainingValidationDataset(valid_x, valid_y, transform=valid_transform, features=features)

        sampler = None
        print("Train", train_ds)
        print("Valid", valid_ds)
        return train_ds, valid_ds, sampler


def get_datasets_batched(
    data_dir: str,
    fold: int,
    augmentation: str = "light",
    fast: bool = False,
    image_size: Tuple[int, int] = (512, 512),
    features=None,
):
    train_transform = get_augmentations(augmentation, image_size)
    valid_transform = A.NoOp()

    if fold is None:
        if fast:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

            class_0 = fs.find_images_in_dir(os.path.join(data_dir, "Cover"))

            sampler = WeightedRandomSampler(np.ones(len(class_0)), 2048)

            train_ds = BatchedImageDataset(class_0, transform=train_transform, features=features)
            valid_ds = BatchedImageDataset(class_0, transform=valid_transform, features=features)

            print("Train", train_ds)
            print("Valid", valid_ds)

            return train_ds, valid_ds, sampler
        else:
            raise ValueError("Fold must be set")
    else:
        data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds.csv"))

        train_images = data_folds.loc[INPUT_IMAGE_ID_KEY, data_folds[INPUT_FOLD_KEY] != fold].tolist()
        valid_images = data_folds.loc[INPUT_IMAGE_ID_KEY, data_folds[INPUT_FOLD_KEY] == fold].tolist()

        train_images = [os.path.join(data_dir, "Cover", x) for x in train_images]
        valid_images = [os.path.join(data_dir, "Cover", x) for x in valid_images]

        train_ds = BatchedImageDataset(train_images, transform=train_transform, features=features)
        valid_ds = BatchedImageDataset(valid_images, transform=valid_transform, features=features)

        sampler = None
        print("Train", train_ds)
        print("Valid", valid_ds)
        return train_ds, valid_ds, sampler


def get_test_dataset(data_dir, features):
    valid_transform = A.NoOp()
    images = fs.find_images_in_dir(os.path.join(data_dir, "Test"))
    return TrainingValidationDataset(images, None, valid_transform, features=features)
