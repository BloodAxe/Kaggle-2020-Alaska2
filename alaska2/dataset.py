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
INPUT_FEATURES_ELA_RICH_KEY = "input_ela_rich"
INPUT_FEATURES_BLUR_KEY = "input_blur"
INPUT_IMAGE_ID_KEY = "image_id"
INPUT_IMAGE_QF_KEY = "image_qf"
INPUT_FOLD_KEY = "fold"

HOLDOUT_FOLD = -1

INPUT_FEATURES_DCT_KEY = "input_dct"
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
    "INPUT_FEATURES_CHANNEL_CB_KEY",
    "INPUT_FEATURES_CHANNEL_CR_KEY",
    "INPUT_FEATURES_CHANNEL_Y_KEY",
    "INPUT_FEATURES_DCT_CB_KEY",
    "INPUT_FEATURES_DCT_CR_KEY",
    "INPUT_FEATURES_DCT_KEY",
    "INPUT_FEATURES_DCT_Y_KEY",
    "INPUT_FEATURES_ELA_KEY",
    "INPUT_FEATURES_ELA_RICH_KEY",
    "INPUT_FOLD_KEY",
    "INPUT_IMAGE_ID_KEY",
    "INPUT_IMAGE_KEY",
    "INPUT_IMAGE_QF_KEY",
    "INPUT_TRUE_MODIFICATION_FLAG",
    "INPUT_TRUE_MODIFICATION_TYPE",
    "OUTPUT_FEATURE_MAP_16",
    "OUTPUT_FEATURE_MAP_32",
    "OUTPUT_FEATURE_MAP_4",
    "OUTPUT_FEATURE_MAP_8",
    "OUTPUT_PRED_EMBEDDING",
    "OUTPUT_PRED_MODIFICATION_FLAG",
    "OUTPUT_PRED_MODIFICATION_TYPE",
    "PairedImageDataset",
    "QuadImageDataset",
    "TrainingValidationDataset",
    "compute_blur_features",
    "compute_dct_fast",
    "compute_dct_slow",
    "compute_ela",
    "compute_features",
    "dct8",
    "get_datasets",
    "get_datasets_paired",
    "get_datasets_quad",
    "get_test_dataset",
    "idct8",
    "get_holdout",
    "get_negatives_ds",
]


def compute_dct_fast(jpeg_file):
    from jpeg2dct.numpy import load, loads

    dct_y, dct_cb, dct_cr = load(jpeg_file)
    return dct_y, dct_cb, dct_cr


def get_dctmtx():
    [Col, Row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * Col + 1) * Row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    return T


# DCTMTX = np.array(
#     [
#         [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
#         [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
#         [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
#         [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
#         [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
#         [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
#         [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
#         [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975],
#     ],
#     dtype=np.float32,
# )

DCTMTX = get_dctmtx()


def compute_dct_slow(jpeg_file):
    image = cv2.imread(jpeg_file)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(ycrcb)
    return dct8(y), dct8(cr), dct8(cb)


def dct2channels_last(image):
    """
    Rearrange DCT image from [H,W] to [H//8, W//8, 64]
    """
    assert len(image.shape) == 2, f"{image.shape}"
    assert image.shape[0] % 8 == 0, f"{image.shape}"
    assert image.shape[1] % 8 == 0, f"{image.shape}"

    block_view = (image.shape[0] // 8, 8, image.shape[1] // 8, 8)
    dct_shape = (image.shape[0] // 8, image.shape[1] // 8, 64)
    block_permute = 0, 2, 1, 3

    result = image.reshape(block_view).transpose(*block_permute).reshape(dct_shape)
    return result


def dct2spatial(image):
    """
    Rearrange DCT image from [H//8, W//8, 64] to [H,W]
    """
    assert image.shape[2] == 64

    block_view = (image.shape[0], image.shape[1], 8, 8)
    image_shape = (image.shape[0] * 8, image.shape[1] * 8)
    block_permute = 0, 2, 1, 3

    result = image.reshape(block_view).transpose(*block_permute).reshape(image_shape)
    return result


def dct8(image):
    assert image.shape[0] % 8 == 0
    assert image.shape[1] % 8 == 0
    dct_shape = (image.shape[0] // 8, image.shape[1] // 8, 64)
    dct_image = np.zeros(dct_shape, dtype=np.float32)

    one_over_255 = np.float32(1.0 / 255.0)
    image = image * one_over_255
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            # dct = cv2.dct(image[i : i + 8, j : j + 8])
            dct = DCTMTX @ image[i : i + 8, j : j + 8] @ DCTMTX.T
            dct_image[i // 8, j // 8, :] = dct.flatten()

    return dct_image


def idct8(dct):
    assert dct.shape[2] == 64
    dct_image = np.zeros((dct.shape[0] * 8, dct.shape[1] * 8), dtype=np.float32)

    for i in range(0, dct.shape[0]):
        for j in range(0, dct.shape[1]):
            img = DCTMTX.T @ dct[i, j].reshape((8, 8)) @ DCTMTX
            dct_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = img

    return dct_image


def idct8v2(dct, qm=None):
    decoded_image = np.zeros((dct.shape[0], dct.shape[1], 1), dtype=np.float32)

    if qm is None:
        for i in range(0, dct.shape[0] // 8):
            for j in range(0, dct.shape[1] // 8):
                img = DCTMTX.T @ (dct[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]) @ DCTMTX
                decoded_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, 0] = img
    else:
        for i in range(0, dct.shape[0] // 8):
            for j in range(0, dct.shape[1] // 8):
                img = DCTMTX.T @ (qm * dct[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]) @ DCTMTX
                decoded_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8, 0] = img

    return decoded_image


def compute_ela(image, quality_steps=[75]):
    diff = np.zeros((image.shape[0], image.shape[1], 3 * len(quality_steps)), dtype=np.float32)

    for i, q in enumerate(quality_steps):
        retval, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
        image_lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        np.subtract(image_lq, image, out=diff[..., i * 3 : i * 3 + 3], dtype=np.float32)

    return diff


def compute_ela_rich(image, quality_steps=[75, 99, 100]):
    diff = np.zeros((image.shape[0], image.shape[1], len(quality_steps)), dtype=np.float32)

    for i, q in enumerate(quality_steps):
        retval, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
        image_lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        diff[..., i] = np.abs(np.subtract(image_lq, image, dtype=np.float32)).sum(axis=2)

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

    if INPUT_FEATURES_ELA_RICH_KEY in features:
        sample[INPUT_FEATURES_ELA_RICH_KEY] = compute_ela_rich(image, quality_steps=[75, 95, 99, 100])

    if INPUT_FEATURES_BLUR_KEY in features:
        sample[INPUT_FEATURES_BLUR_KEY] = compute_blur_features(image)

    if INPUT_FEATURES_DCT_KEY in features:
        dct_file = np.load(fs.change_extension(image_fname, ".npz"))
        dct_y, dct_cb, dct_cr = dct_file["dct_y"], dct_file["dct_cb"], dct_file["dct_cr"]
        sample[INPUT_FEATURES_DCT_KEY] = np.dstack([dct_y, dct_cb, dct_cr])

    if INPUT_FEATURES_DCT_Y_KEY in features:
        dct_file = np.load(fs.change_extension(image_fname, ".npz"))
        sample[INPUT_FEATURES_DCT_Y_KEY] = dct_file["dct_y"]
        sample[INPUT_FEATURES_DCT_CB_KEY] = dct_file["dct_cb"]
        sample[INPUT_FEATURES_DCT_CR_KEY] = dct_file["dct_cr"]

    if INPUT_FEATURES_CHANNEL_Y_KEY in features:
        dct_file = np.load(fs.change_extension(image_fname, ".npz"))
        # This normalization roughly puts values into zero mean and unit variance
        sample[INPUT_FEATURES_CHANNEL_Y_KEY] = idct8v2(dct_file["dct_y"])
        sample[INPUT_FEATURES_CHANNEL_CB_KEY] = idct8v2(dct_file["dct_cb"])
        sample[INPUT_FEATURES_CHANNEL_CR_KEY] = idct8v2(dct_file["dct_cr"])

    return sample


class TrainingValidationDataset(Dataset):
    def __init__(
        self,
        images: Union[List, np.ndarray],
        targets: Optional[Union[List, np.ndarray]],
        quality: Union[List, np.ndarray],
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
        self.quality = quality

        self.obliterate = obliterate
        self.obliterate_p = obliterate_p

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"TrainingValidationDataset(len={len(self)}, targets_hist={np.bincount(self.targets)}, features={self.features})"

    def __getitem__(self, index):
        image_fname = self.images[index]
        image = cv2.imread(image_fname)
        qf = self.quality[index]
        data = {}
        data["image"] = image
        data.update(compute_features(image, image_fname, self.features))

        data = self.transform(**data)

        sample = {INPUT_IMAGE_ID_KEY: os.path.basename(self.images[index]), INPUT_IMAGE_QF_KEY: int(qf)}

        if self.targets is not None:
            target = int(self.targets[index])

            if (self.obliterate_p > 0) and (random.random() < self.obliterate_p):
                target = 0
                data = self.obliterate(**data)

            sample[INPUT_TRUE_MODIFICATION_TYPE] = target
            sample[INPUT_TRUE_MODIFICATION_FLAG] = torch.tensor([target > 0]).float()

        for key, value in data.items():
            if key in self.features:
                sample[key] = tensor_from_rgb_image(value)

        return sample


class PairedImageDataset(Dataset):
    def __init__(self, images: Union[np.ndarray, List], target: int, transform: A.Compose, features):
        self.images = images
        self.features = features
        self.target = target
        self.method_name = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]

        assert 0 < target < 4
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"PairedImageDataset(images={len(self.images)})"

    def __getitem__(self, index):
        cover_image_fname = self.images[index]

        method_name = self.method_name[self.target]
        secret_image_fname = cover_image_fname.replace("Cover", method_name)

        cover_image = cv2.imread(cover_image_fname)
        secret_image = cv2.imread(secret_image_fname)

        cover_data = {}
        cover_data["image"] = cover_image
        cover_data.update(compute_features(cover_image, cover_image_fname, self.features))
        cover_data = self.transform(**cover_data)

        secret_data = {}
        secret_data["image"] = secret_image
        secret_data.update(compute_features(secret_image, secret_image_fname, self.features))
        secret_data = self.transform(**secret_data)

        sample = {
            INPUT_IMAGE_ID_KEY: [fs.id_from_fname(cover_image_fname), fs.id_from_fname(secret_image_fname)],
            INPUT_TRUE_MODIFICATION_TYPE: torch.tensor([0, self.target]).long(),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([0, 1]).float(),
        }

        for key, value in cover_data.items():
            if key in self.features:
                sample[key] = torch.stack(
                    [tensor_from_rgb_image(cover_data[key]), tensor_from_rgb_image(secret_data[key])]
                )

        return sample


class QuadImageDataset(Dataset):
    def __init__(self, images: Union[np.ndarray, List], transform: A.ReplayCompose, features, use_replay=False):
        self.images = images
        self.features = features
        self.transform = transform
        self.use_replay = use_replay

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"QuadImageDataset(images={len(self.images)}, use_replay={self.use_replay})"

    def __getitem__(self, index):
        class0_fname = self.images[index]
        class1_fname = class0_fname.replace("Cover", "JMiPOD")
        class2_fname = class0_fname.replace("Cover", "JUNIWARD")
        class3_fname = class0_fname.replace("Cover", "UERD")

        class0_image = cv2.imread(class0_fname)
        class1_image = cv2.imread(class1_fname)
        class2_image = cv2.imread(class2_fname)
        class3_image = cv2.imread(class3_fname)

        class0_data = {}
        class0_data["image"] = class0_image
        class0_data.update(compute_features(class0_image, class0_fname, self.features))
        class0_data = self.transform(**class0_data)

        replay = class0_data["replay"]

        class1_data = {}
        class1_data["image"] = class1_image
        class1_data.update(compute_features(class1_image, class1_fname, self.features))
        class1_data = (
            self.transform.replay(replay, **class1_data) if self.use_replay else self.transform(**class1_data)
        )

        class2_data = {}
        class2_data["image"] = class2_image
        class2_data.update(compute_features(class2_image, class2_fname, self.features))
        class2_data = (
            self.transform.replay(replay, **class2_data) if self.use_replay else self.transform(**class2_data)
        )

        class3_data = {}
        class3_data["image"] = class3_image
        class3_data.update(compute_features(class3_image, class3_fname, self.features))
        class3_data = (
            self.transform.replay(replay, **class3_data) if self.use_replay else self.transform(**class3_data)
        )

        sample = {
            INPUT_IMAGE_ID_KEY: [
                fs.id_from_fname(class0_fname),
                fs.id_from_fname(class1_fname),
                fs.id_from_fname(class2_fname),
                fs.id_from_fname(class3_fname),
            ],
            INPUT_TRUE_MODIFICATION_TYPE: torch.tensor([0, 1, 2, 3]).long(),
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([0, 1, 1, 1]).float(),
        }

        for key, value in class0_data.items():
            if key in self.features:
                sample[key] = torch.stack(
                    [
                        tensor_from_rgb_image(class0_data[key]),
                        tensor_from_rgb_image(class1_data[key]),
                        tensor_from_rgb_image(class2_data[key]),
                        tensor_from_rgb_image(class3_data[key]),
                    ]
                )

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
        data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds_v2.csv"))
        unchanged = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "df_unchanged.csv"))

        # Ignore holdout fold
        data_folds = data_folds[data_folds[INPUT_FOLD_KEY] != HOLDOUT_FOLD]

        train_df = data_folds[data_folds[INPUT_FOLD_KEY] != fold]
        valid_df = data_folds[data_folds[INPUT_FOLD_KEY] == fold]

        if fast:
            train_df = train_df[::50]
            valid_df = valid_df[::50]

        train_images = train_df[INPUT_IMAGE_ID_KEY].tolist()
        valid_images = valid_df[INPUT_IMAGE_ID_KEY].tolist()

        train_images = [os.path.join(data_dir, "Cover", x) for x in train_images]
        valid_images = [os.path.join(data_dir, "Cover", x) for x in valid_images]

        train_x = train_images.copy()
        train_y = [0] * len(train_images)
        train_qf = train_df["quality"].values.tolist()

        valid_x = valid_images.copy()
        valid_y = [0] * len(valid_images)
        valid_qf = valid_df["quality"].values.tolist()

        for method_index, method_name in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
            # Filter images that does not have any alterations DCT (there are 250 of them)
            unchanged_files = unchanged[unchanged["method"] == method_name].file.values
            unchanged_files = list(map(fs.id_from_fname, unchanged_files))

            for fname, qf in zip(train_images, train_df["quality"].values):
                if fs.id_from_fname(fname) not in unchanged_files:
                    fname = fname.replace("Cover", method_name)
                    train_x.append(fname)
                    train_y.append(method_index + 1)
                    train_qf.append(qf)
                else:
                    print("Removed unchanged file from the train set", fname)

            for fname, qf in zip(valid_images, valid_df["quality"].values):
                if fs.id_from_fname(fname) not in unchanged_files:
                    fname = fname.replace("Cover", method_name)
                    valid_x.append(fname)
                    valid_y.append(method_index + 1)
                    valid_qf.append(qf)
                else:
                    print("Removed unchanged file from the valid set", fname)

            # train_x += [fname.replace("Cover", method) for fname in train_images]
            # train_y += [i + 1] * len(train_images)
            # train_qf += train_df["quality"].values.tolist()

            # valid_x += [fname.replace("Cover", method) for fname in valid_images]
            # valid_y += [i + 1] * len(valid_images)
            # valid_qf += valid_df["quality"].values.tolist()

        assert len(set(train_x).intersection(set(valid_x))) == 0, "Train set and valid set has common elements"

        train_ds = TrainingValidationDataset(
            images=train_x,
            targets=train_y,
            quality=train_qf,
            transform=train_transform,
            features=features,
            obliterate=get_obliterate_augs() if obliterate_p > 0 else None,
            obliterate_p=obliterate_p,
        )
        valid_ds = TrainingValidationDataset(
            images=valid_x, targets=valid_y, quality=valid_qf, transform=valid_transform, features=features
        )

        sampler = None
        print("Train", train_ds)
        print("Valid", valid_ds)
        return train_ds, valid_ds, sampler


def get_holdout(data_dir: str, image_size: Tuple[int, int] = (512, 512), features=None):
    valid_transform = A.NoOp()

    data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds_v2.csv"))
    unchanged = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "df_unchanged.csv"))

    # Take only holdout fold
    holdout_df = data_folds[data_folds[INPUT_FOLD_KEY] == HOLDOUT_FOLD]

    holdout_images = holdout_df[INPUT_IMAGE_ID_KEY].tolist()
    holdout_images = [os.path.join(data_dir, "Cover", x) for x in holdout_images]

    valid_x = holdout_images.copy()
    valid_y = [0] * len(holdout_images)
    valid_qf = holdout_df["quality"].values.tolist()

    for method_index, method_name in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
        # Filter images that does not have any alterations DCT (there are 250 of them)
        unchanged_files = unchanged[unchanged["method"] == method_name].file.values
        unchanged_files = list(map(fs.id_from_fname, unchanged_files))

        for fname, qf in zip(holdout_images, holdout_df["quality"].values):
            if fs.id_from_fname(fname) not in unchanged_files:
                fname = fname.replace("Cover", method_name)
                valid_x.append(fname)
                valid_y.append(method_index + 1)
                valid_qf.append(qf)
            else:
                print("Removed unchanged file from the holdout set", fname)

    holdout_ds = TrainingValidationDataset(
        images=valid_x, targets=valid_y, quality=valid_qf, transform=valid_transform, features=features
    )

    print("Holdout", holdout_ds)
    return holdout_ds


def get_negatives_ds(data_dir, features, fold: int, local_rank=0, image_size=(512, 512), max_images=None):
    negative_images = fs.find_images_in_dir(data_dir)

    if max_images is not None:
        negative_images = negative_images[:max_images]

    return TrainingValidationDataset(
        images=negative_images,
        targets=[0] * len(negative_images),
        quality=[0] * len(negative_images),
        transform=A.Compose(
            [
                A.Transpose(p=0.5),
                A.RandomRotate90(p=1.0),
                A.ShiftScaleRotate(),
                A.RandomBrightnessContrast(),
                A.LongestMaxSize(max(*image_size)),
                A.PadIfNeeded(image_size[0], image_size[0]),
                A.ImageCompression(quality_lower=75, quality_upper=95),
            ]
        ),
        features=features,
    )


def get_datasets_paired(
    data_dir: str,
    fold: int,
    augmentation: str = "light",
    fast: bool = False,
    image_size: Tuple[int, int] = (512, 512),
    features=None,
):
    from .augmentations import get_augmentations

    train_transform = get_augmentations(augmentation, image_size)
    valid_transform = A.ReplayCompose([A.NoOp()])

    data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds.csv"))

    # Ignore holdout fold
    data_folds = data_folds[data_folds[INPUT_FOLD_KEY] != HOLDOUT_FOLD]

    train_images = data_folds.loc[data_folds[INPUT_FOLD_KEY] != fold, INPUT_IMAGE_ID_KEY].tolist()
    train_images = [os.path.join(data_dir, "Cover", x) for x in train_images]

    train_ds = (
        PairedImageDataset(train_images, target=1, transform=train_transform, features=features)
        + PairedImageDataset(train_images, target=2, transform=train_transform, features=features)
        + PairedImageDataset(train_images, target=3, transform=train_transform, features=features)
    )

    valid_images = data_folds.loc[data_folds[INPUT_FOLD_KEY] == fold, INPUT_IMAGE_ID_KEY].tolist()
    valid_images = [os.path.join(data_dir, "Cover", x) for x in valid_images]

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


def get_datasets_quad(
    data_dir: str,
    fold: int,
    augmentation: str = "light",
    fast: bool = False,
    image_size: Tuple[int, int] = (512, 512),
    features=None,
    use_replay=False,
):
    from .augmentations import get_augmentations

    train_transform = get_augmentations(augmentation, image_size)
    valid_transform = A.NoOp()

    data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds.csv"))

    # Ignore holdout fold
    data_folds = data_folds[data_folds[INPUT_FOLD_KEY] != HOLDOUT_FOLD]

    train_images = data_folds.loc[data_folds[INPUT_FOLD_KEY] != fold, INPUT_IMAGE_ID_KEY].tolist()
    train_images = [os.path.join(data_dir, "Cover", x) for x in train_images]

    train_ds = QuadImageDataset(train_images, transform=train_transform, features=features, use_replay=use_replay)

    valid_images = data_folds.loc[data_folds[INPUT_FOLD_KEY] == fold, INPUT_IMAGE_ID_KEY].tolist()
    valid_images = [os.path.join(data_dir, "Cover", x) for x in valid_images]

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
    # images = fs.find_images_in_dir(os.path.join(data_dir, "Test"))
    test_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_dataset_qf_qt.csv"))
    test_df["image_fname"] = test_df[""].apply(lambda x: os.path.join(data_dir, "Test", x))

    return TrainingValidationDataset(
        images=test_df["image_fname"].values.tolist(),
        targets=None,
        quality=test_df["quality"].values.tolist(),
        transform=valid_transform,
        features=features,
    )
