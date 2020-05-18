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


INPUT_IMAGE_KEY = "image"
INPUT_FEATURES_ELA_KEY = "input_ela"
INPUT_FEATURES_BLUR_KEY = "input_blur"
INPUT_IMAGE_ID_KEY = "image_id"
INPUT_FOLD_KEY = "fold"

INPUT_FEATURES_DCT_Y_KEY = "input_dct_y"
INPUT_FEATURES_DCT_CR_KEY = "input_dct_cr"
INPUT_FEATURES_DCT_CB_KEY = "input_dct_cb"

INPUT_FEATURES_CHANNEL_Y_KEY = "input_image_y"
INPUT_FEATURES_CHANNEL_CR_KEY = "input_image_cr"
INPUT_FEATURES_CHANNEL_CB_KEY = "input_image_cb"

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
    "INPUT_FEATURES_DCT_CB_KEY",
    "INPUT_FEATURES_DCT_CR_KEY",
    "INPUT_FEATURES_DCT_Y_KEY",
    "INPUT_FEATURES_ELA_KEY",
    "INPUT_FEATURES_CHANNEL_Y_KEY",
    "INPUT_FEATURES_CHANNEL_CB_KEY",
    "INPUT_FEATURES_CHANNEL_CR_KEY",
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
    "compute_dct_fast",
    "compute_dct_slow",
    "compute_ela",
    "dct8",
    "idct8",
    "get_datasets",
    "get_datasets_batched",
    "get_test_dataset",
]


def compute_dct_fast(jpeg_file):
    from jpeg2dct.numpy import load, loads

    dct_y, dct_cb, dct_cr = load(jpeg_file)
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


def compute_dct_slow(jpeg_file):
    image = cv2.imread(jpeg_file)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    cr = cv2.pyrDown(cr)
    cb = cv2.pyrDown(cb)
    return dct8(y), dct8(cr), dct8(cb)


def dct8(image):
    assert image.shape[0] % 8 == 0
    assert image.shape[1] % 8 == 0
    dct_image = np.zeros((image.shape[0] // 8, image.shape[1] // 8, 64), dtype=np.float32)

    one_over_255 = np.float32(1.0 / 255.0)
    image = image * one_over_255
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            # dct = cv2.dct(image[i : i + 8, j : j + 8])
            dct = DCTMTX @ image[i : i + 8, j : j + 8] @ DCTMTX.T
            dct_image[i // 8, j // 8, :] = dct.flatten()

    return dct_image


def idct8(dct):
    dct_image = np.zeros((dct.shape[0] * 8, dct.shape[1] * 8, 1), dtype=np.float32)

    for i in range(0, dct.shape[0]):
        for j in range(0, dct.shape[1]):
            img = DCTMTX.T @ dct[i, j].reshape((8, 8)) @ DCTMTX
            dct_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, 0] = img

    return dct_image


def compute_ela(image, quality_steps=[75]):
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


def compute_features(image: np.ndarray, image_fname: str, features):
    sample = {}

    if INPUT_FEATURES_ELA_KEY in features:
        sample[INPUT_FEATURES_ELA_KEY] = compute_ela(image)

    if INPUT_FEATURES_BLUR_KEY in features:
        sample[INPUT_FEATURES_BLUR_KEY] = compute_blur_features(image)

    if INPUT_FEATURES_CHANNEL_Y_KEY in features:
        dct_file = np.load(fs.change_extension(image_fname, ".npz"))
        # This normalization roughly puts values into zero mean and unit variance
        sample[INPUT_FEATURES_CHANNEL_Y_KEY] = idct8(dct_file["dct_y"])
        sample[INPUT_FEATURES_CHANNEL_CR_KEY] = idct8(dct_file["dct_cr"])
        sample[INPUT_FEATURES_CHANNEL_CB_KEY] = idct8(dct_file["dct_cb"])

    return sample


class TrainingValidationDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        targets: Optional[Union[List, np.ndarray]],
        transform: A.Compose,
        features: List[str],
        obliterate: A.Compose = None,
        obliterate_p=0.0,
    ):
        """
        :param obliterate - Augmentation that destroys embedding.
        """
        if targets is not None:
            if len(images) != len(targets):
                raise ValueError(f"Size of images and targets does not match: {len(images)} {len(targets)}")

        self.images = images
        self.targets = targets
        self.transform = transform
        self.features = features

        self.obliterate = obliterate
        self.obliterate_p = obliterate_p

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"TrainingValidationDataset(len={len(self)}, targets_hist={np.bincount(self.targets)}, features={self.features})"

    def __getitem__(self, index):
        image_fname = self.images[index]
        image = cv2.imread(image_fname)

        data = {}
        data["image"] = image
        data.update(compute_features(image, image_fname, self.features))

        data = self.transform(**data)

        sample = {INPUT_IMAGE_ID_KEY: fs.id_from_fname(self.images[index])}

        if self.targets is not None:
            t = self.targets[index]

            if t > 0 and self.obliterate_p > 0 and self.obliterate_p > random.random():
                t = 0
                data = self.obliterate(**data)

            sample[INPUT_TRUE_MODIFICATION_TYPE] = int(t)
            sample[INPUT_TRUE_MODIFICATION_FLAG] = torch.tensor([t > 0]).float()

        for key, value in data.items():
            if key in self.features:
                sample[key] = tensor_from_rgb_image(value)

        return sample


class PairedImageDataset(Dataset):
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
        image_fname0 = self.images[index]
        image_fname1 = image_fname0.replace("Cover", "JMiPOD")
        image_fname2 = image_fname0.replace("Cover", "JUNIWARD")
        image_fname3 = image_fname0.replace("Cover", "UERD")

        method = random.randint(0, 2)

        modified_fname = [image_fname1, image_fname2, image_fname3][method]
        image0 = cv2.imread(image_fname0)
        image1 = cv2.imread(modified_fname)

        data = self.transform(image=image0)

        image0 = data["image"]
        image1 = self.transform.replay(data["replay"], image=image1)["image"]

        sample = {
            INPUT_IMAGE_ID_KEY: [fs.id_from_fname(image_fname0), fs.id_from_fname(modified_fname)],
            INPUT_IMAGE_KEY: torch.stack([tensor_from_rgb_image(image0), tensor_from_rgb_image(image1)]),
            INPUT_TRUE_MODIFICATION_TYPE: torch.tensor([0, method + 1]).long(),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([0, 1]).float(),
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
    obliterate_p=0.0,
):
    from .augmentations import get_augmentations, get_obliterate_augs

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

        if fast:
            train_images = train_images[::10]
            valid_images = valid_images[::10]

        train_x = train_images.copy()
        train_y = [0] * len(train_images)

        valid_x = valid_images.copy()
        valid_y = [0] * len(valid_images)

        for i, method in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
            train_x += [fname.replace("Cover", method) for fname in train_images]
            train_y += [i + 1] * len(train_images)

            valid_x += [fname.replace("Cover", method) for fname in valid_images]
            valid_y += [i + 1] * len(valid_images)

        train_ds = TrainingValidationDataset(
            train_x,
            train_y,
            transform=train_transform,
            features=features,
            obliterate=get_obliterate_augs() if obliterate_p > 0 else None,
            obliterate_p=obliterate_p,
        )
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
    from .augmentations import get_augmentations

    train_transform = get_augmentations(augmentation, image_size)
    valid_transform = A.NoOp()

    if fold is None:
        if fast:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

            class_0 = fs.find_images_in_dir(os.path.join(data_dir, "Cover"))

            sampler = WeightedRandomSampler(np.ones(len(class_0)), 2048)

            train_ds = PairedImageDataset(class_0, transform=train_transform, features=features)
            valid_ds = PairedImageDataset(class_0, transform=valid_transform, features=features)

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

        if fast:
            train_images = train_images[::10]
            valid_images = valid_images[::10]

        train_ds = PairedImageDataset(train_images, transform=train_transform, features=features)

        valid_x = valid_images.copy()
        valid_y = [0] * len(valid_images)

        for i, method in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
            valid_x += [fname.replace("Cover", method) for fname in valid_images]
            valid_y += [i + 1] * len(valid_images)
        valid_ds = TrainingValidationDataset(valid_x, valid_y, transform=valid_transform, features=features)

        sampler = None
        print("Train", train_ds)
        print("Valid", valid_ds)
        return train_ds, valid_ds, sampler


def get_test_dataset(data_dir, features):
    valid_transform = A.NoOp()
    images = fs.find_images_in_dir(os.path.join(data_dir, "Test"))
    return TrainingValidationDataset(images, None, valid_transform, features=features)
