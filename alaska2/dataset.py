import json
import math
import os
import random
from typing import Tuple, Optional, Union, List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset, ConcatDataset

INPUT_IMAGE_KEY = "image"
INPUT_FEATURES_ELA_KEY = "input_ela"
INPUT_FEATURES_ELA_RICH_KEY = "input_ela_rich"
INPUT_FEATURES_JPEG_FLOAT = "input_raw_jpeg"
INPUT_FEATURES_BLUR_KEY = "input_blur"
INPUT_FEATURES_DECODING_RESIDUAL_KEY = "input_residual"
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
INPUT_TRUE_MODIFICATION_MASK = "true_embedding_mask"
INPUT_TRUE_PAYLOAD_BITS = "true_payload_bits"

OUTPUT_PRED_MODIFICATION_TYPE = "pred_modification_type"
OUTPUT_PRED_MODIFICATION_FLAG = "pred_modification_flag"
OUTPUT_PRED_MODIFICATION_MASK = "pred_embedding_mask"
OUTPUT_PRED_PAYLOAD_BITS = "pred_payload_bits"

OUTPUT_PRED_EMBEDDING = "pred_embedding"

OUTPUT_FEATURE_MAP_4 = "pred_fm_4"
OUTPUT_FEATURE_MAP_8 = "pred_fm_8"
OUTPUT_FEATURE_MAP_16 = "pred_fm_16"
OUTPUT_FEATURE_MAP_32 = "pred_fm_32"

METHOD_TO_INDEX = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 2, "j_uniward": 2, "UERD": 3, "uerd": 3, "nsf5": 4}

__all__ = [
    "HOLDOUT_FOLD",
    "INPUT_FEATURES_BLUR_KEY",
    "INPUT_FEATURES_CHANNEL_CB_KEY",
    "INPUT_FEATURES_CHANNEL_CR_KEY",
    "INPUT_FEATURES_CHANNEL_Y_KEY",
    "INPUT_FEATURES_DCT_CB_KEY",
    "INPUT_FEATURES_DCT_CR_KEY",
    "INPUT_FEATURES_DCT_KEY",
    "INPUT_FEATURES_DCT_Y_KEY",
    "INPUT_FEATURES_DECODING_RESIDUAL_KEY",
    "INPUT_FEATURES_ELA_KEY",
    "INPUT_FEATURES_ELA_RICH_KEY",
    "INPUT_FEATURES_JPEG_FLOAT",
    "INPUT_FOLD_KEY",
    "INPUT_IMAGE_ID_KEY",
    "INPUT_IMAGE_KEY",
    "INPUT_IMAGE_QF_KEY",
    "INPUT_TRUE_MODIFICATION_FLAG",
    "INPUT_TRUE_MODIFICATION_MASK",
    "INPUT_TRUE_MODIFICATION_TYPE",
    "INPUT_TRUE_PAYLOAD_BITS",
    "OUTPUT_FEATURE_MAP_16",
    "OUTPUT_FEATURE_MAP_32",
    "OUTPUT_FEATURE_MAP_4",
    "OUTPUT_FEATURE_MAP_8",
    "OUTPUT_PRED_EMBEDDING",
    "OUTPUT_PRED_MODIFICATION_FLAG",
    "OUTPUT_PRED_MODIFICATION_MASK",
    "OUTPUT_PRED_MODIFICATION_TYPE",
    "OUTPUT_PRED_PAYLOAD_BITS",
    "PairedImageDataset",
    "TrainingValidationDataset",
    "bitmix",
    "compute_blur_features",
    "compute_dct_fast",
    "compute_dct_slow",
    "compute_ela",
    "compute_features",
    "dct8",
    "get_datasets",
    "get_datasets_paired",
    "get_holdout",
    "get_istego100k_test_other",
    "get_istego100k_test_same",
    "get_istego100k_train",
    "get_negatives_ds",
    "get_test_dataset",
    "idct8",
]


def compute_dct_fast(jpeg_file):
    from jpeg2dct.numpy import load

    dct_y, dct_cb, dct_cr = load(jpeg_file)
    return dct_y, dct_cb, dct_cr


def get_dctmtx():
    [Col, Row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * Col + 1) * Row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    return T.astype(np.float32)


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


def decode_bgr_from_dct(dct_file):
    dct = np.load(dct_file)
    dct_y = dct["dct_y"]
    dct_cr = dct["dct_cr"]
    dct_cb = dct["dct_cb"]

    y = idct8v2(dct_y)
    cr = idct8v2(dct_cr)
    cb = idct8v2(dct_cb)

    y += 127.5
    y /= 255.0

    cr += 127.5
    cr /= 255.0

    cb += 127.5
    cb /= 255.0

    img_ycrcb = np.dstack([y, cr, cb])
    bgr_from_dct = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    return bgr_from_dct


def compute_decoding_residual(image: np.ndarray, dct_fname: str) -> np.ndarray:
    bgr_from_dct = decode_bgr_from_dct(dct_fname)
    return bgr_from_dct * 255 - image.astype(np.float32)


def compute_ela(image, quality_steps=[75]):
    diff = np.zeros((image.shape[0], image.shape[1], 3 * len(quality_steps)), dtype=np.float32)

    for i, q in enumerate(quality_steps):
        retval, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
        image_lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        np.subtract(image_lq, image, out=diff[..., i * 3 : i * 3 + 3], dtype=np.float32)

    return diff


def compute_ela_rich(image, quality_steps=[75, 90, 95]):
    diff = np.zeros((image.shape[0], image.shape[1], len(quality_steps) * 3), dtype=np.float32)

    for i, q in enumerate(quality_steps):
        retval, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
        image_lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        diff[..., i * 3 : i * 3 + 3] = np.abs(np.subtract(image_lq, image, dtype=np.float32))

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

    if INPUT_TRUE_MODIFICATION_MASK in features:
        mask = fs.change_extension(image_fname, ".png")
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        sample[INPUT_TRUE_MODIFICATION_MASK] = mask

    if INPUT_FEATURES_ELA_RICH_KEY in features:
        sample[INPUT_FEATURES_ELA_RICH_KEY] = compute_ela_rich(image, quality_steps=[75, 90, 95])

    if INPUT_FEATURES_BLUR_KEY in features:
        sample[INPUT_FEATURES_BLUR_KEY] = compute_blur_features(image)

    if INPUT_FEATURES_JPEG_FLOAT in features:
        dct_file = fs.change_extension(image_fname, ".npz")
        sample[INPUT_FEATURES_JPEG_FLOAT] = 255 * decode_bgr_from_dct(dct_file)

    if INPUT_FEATURES_DECODING_RESIDUAL_KEY in features:
        dct_file = fs.change_extension(image_fname, ".npz")
        sample[INPUT_FEATURES_DECODING_RESIDUAL_KEY] = compute_decoding_residual(image, dct_file)

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


def block8_sum(a: np.ndarray):
    shape = a.shape
    a = a.reshape([a.shape[0] // 8, 8, a.shape[1] // 8, 8] + list(a.shape[2:]))
    a = a.sum(axis=(1, 3))
    return a


class TrainingValidationDataset(Dataset):
    def __init__(
        self,
        images: Union[List, np.ndarray],
        targets: Optional[Union[List, np.ndarray]],
        quality: Union[List, np.ndarray],
        bits: Optional[Union[List, np.ndarray]],
        transform: Union[A.Compose, A.BasicTransform],
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
        self.bits = bits

        self.obliterate = obliterate
        self.obliterate_p = obliterate_p

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"TrainingValidationDataset(len={len(self)}, targets_hist={np.bincount(self.targets)}, qf={np.bincount(self.quality)}, features={self.features})"

    def __getitem__(self, index):
        image_fname = self.images[index]
        try:
            image = cv2.imread(image_fname)
            if image is None:
                raise FileNotFoundError(image_fname)
        except Exception as e:
            print("Cannot read image ", image_fname, "at index", index)
            print(e)

        qf = self.quality[index]
        data = {}
        data["image"] = image
        data.update(compute_features(image, image_fname, self.features))

        data = self.transform(**data)

        sample = {INPUT_IMAGE_ID_KEY: os.path.basename(self.images[index]), INPUT_IMAGE_QF_KEY: int(qf)}

        if self.bits is not None:
            # OK
            sample[INPUT_TRUE_PAYLOAD_BITS] = torch.tensor(self.bits[index], dtype=torch.float32)

        if self.targets is not None:
            target = int(self.targets[index])

            if (self.obliterate_p > 0) and (random.random() < self.obliterate_p):
                target = 0
                data = self.obliterate(**data)

            sample[INPUT_TRUE_MODIFICATION_TYPE] = target
            sample[INPUT_TRUE_MODIFICATION_FLAG] = torch.tensor([target > 0]).float()

        for key, value in data.items():
            if key in self.features:
                # Mask handling requires some special attention
                if key == INPUT_TRUE_MODIFICATION_MASK:
                    value = np.expand_dims(block8_sum(value) > 0, -1).astype(np.float32)

                sample[key] = tensor_from_rgb_image(value)

        return sample


def bitmix(cover, stego, gamma):
    rows, cols = cover.shape[:2]

    patch_h = int(rows * math.sqrt(gamma))
    patch_w = int(cols * math.sqrt(gamma))

    start_x = random.randint(0, rows - patch_h)
    start_y = random.randint(0, cols - patch_w)
    m = np.zeros_like(cover)
    m[start_y : start_y + patch_h, start_x : start_x + patch_w] = 1

    c = m * stego + (1 - m) * cover
    s = m * cover + (1 - m) * stego

    global_diff = cv2.absdiff(cover, stego)
    local_diff = cv2.absdiff(m * cover, m * stego)
    lam = np.count_nonzero(local_diff) / float(np.count_nonzero(global_diff))
    return c, s, lam, 1 - lam, m


class PairedImageDataset(Dataset):
    def __init__(
        self,
        images: Union[np.ndarray, List],
        quality: List,
        target: int,
        transform: A.ReplayCompose,
        features,
        bitmix=False,
    ):
        self.images = images
        self.features = features
        self.target = target
        self.method_name = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
        self.quality = quality
        self.bitmix = bitmix
        assert 0 < target < 4
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"PairedImageDataset(images={len(self.images)}, target={self.target})"

    def __getitem__(self, index):
        cover_image_fname = self.images[index]

        method_name = self.method_name[self.target]
        stego_image_fname = cover_image_fname.replace("Cover", method_name)

        cover_image = cv2.imread(cover_image_fname)
        stego_image = cv2.imread(stego_image_fname)

        if self.bitmix:
            if random.random() > 0.5:
                try:
                    cover_image, stego_image, cover_target, stego_target, mask = bitmix(
                        cover_image, stego_image, random.uniform(0.25 - 0.03, 0.25 + 0.03)
                    )
                except ZeroDivisionError:
                    print("Bitmix failed due to matching images")
                    print("Number of different pixels", cv2.absdiff(cover_image, stego_image).sum())
                    print(cover_image_fname)
                    print(stego_image_fname)
                    cover_target = 0
                    stego_target = 1

                # NOTE: Type loss is not compatible with bitmix
                type_target = torch.tensor([0, self.target]).long()
                flag_target = torch.tensor([cover_target, stego_target]).float()
            else:
                type_target = torch.tensor([0, self.target]).long()
                flag_target = torch.tensor([0, 1]).float()

        else:
            type_target = torch.tensor([0, self.target]).long()
            flag_target = torch.tensor([0, 1]).float()

        cover_data = {}
        cover_data["image"] = cover_image
        cover_data.update(compute_features(cover_image, cover_image_fname, self.features))
        cover_data = self.transform(**cover_data)

        stego_data = {}
        stego_data["image"] = stego_image
        stego_data.update(compute_features(stego_image, stego_image_fname, self.features))
        stego_data = self.transform.replay(cover_data["replay"], **stego_data)

        qf = int(self.quality[index])

        sample = {
            INPUT_IMAGE_ID_KEY: [fs.id_from_fname(cover_image_fname), fs.id_from_fname(stego_image_fname)],
            INPUT_TRUE_MODIFICATION_TYPE: type_target,
            INPUT_TRUE_MODIFICATION_FLAG: flag_target,
            INPUT_IMAGE_QF_KEY: torch.tensor([qf, qf]),
        }

        # TODO: Add support of mask if this idea will work
        for key, value in cover_data.items():
            if key in self.features:
                sample[key] = torch.stack(
                    [tensor_from_rgb_image(cover_data[key]), tensor_from_rgb_image(stego_data[key])]
                )

        return sample


def get_datasets(
    data_dir: str,
    fold: int,
    augmentation: str = "light",
    fast: bool = False,
    balance=False,
    features=None,
    obliterate_p=0.0,
):
    from .augmentations import get_augmentations, get_obliterate_augs

    train_transform = get_augmentations(augmentation)
    valid_transform = A.NoOp()

    data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds_v2.csv"))
    data_folds["key"] = data_folds["image_id"] + "_" + data_folds["target"].apply(str)

    unchanged = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "df_unchanged.csv"))

    changed_bits = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "analyze_embeddings.csv"))
    changed_bits["key"] = (
        changed_bits["image"] + "_" + changed_bits["method"].apply(lambda x: METHOD_TO_INDEX[x]).apply(str)
    )
    changed_bits_table = {}
    for i, row in changed_bits.iterrows():
        changed_bits_table[row["key"]] = float(row["pd"]) / (512 * 512)

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
    train_bits = [0] * len(train_images)

    valid_x = valid_images.copy()
    valid_y = [0] * len(valid_images)
    valid_qf = valid_df["quality"].values.tolist()
    valid_bits = [0] * len(valid_images)

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

                key = os.path.basename(fname) + "_" + str(method_index)
                train_bits.append(changed_bits_table[key])
            else:
                # print("Removed unchanged file from the train set", fname)
                pass

        for fname, qf in zip(valid_images, valid_df["quality"].values):
            if fs.id_from_fname(fname) not in unchanged_files:
                fname = fname.replace("Cover", method_name)

                valid_x.append(fname)
                valid_y.append(method_index + 1)
                valid_qf.append(qf)

                key = os.path.basename(fname) + "_" + str(method_index)
                valid_bits.append(changed_bits_table[key])
            else:
                # print("Removed unchanged file from the valid set", fname)
                pass

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
        bits=train_bits,
        transform=train_transform,
        features=features,
        obliterate=get_obliterate_augs() if obliterate_p > 0 else None,
        obliterate_p=obliterate_p,
    )
    valid_ds = TrainingValidationDataset(
        images=valid_x,
        targets=valid_y,
        quality=valid_qf,
        bits=valid_bits,
        transform=valid_transform,
        features=features,
    )

    sampler = None
    print("Train", train_ds)
    print("Valid", valid_ds)
    return train_ds, valid_ds, sampler


def get_datasets_paired(
    data_dir: str, fold: int, augmentation: str = "light", bitmix=False, features=None, fast=False
):
    from .augmentations import get_augmentations

    train_transform = get_augmentations(augmentation)
    valid_transform = A.NoOp()

    data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds_v2.csv"))
    unchanged = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "df_unchanged.csv"))

    # Ignore holdout fold
    data_folds = data_folds[data_folds[INPUT_FOLD_KEY] != HOLDOUT_FOLD]

    train_df = data_folds[data_folds[INPUT_FOLD_KEY] != fold]
    valid_df = data_folds[data_folds[INPUT_FOLD_KEY] == fold]

    train_df = train_df[~train_df[INPUT_IMAGE_ID_KEY].isin(unchanged.file)]

    if fast:
        train_df = train_df[::200]
        valid_df = valid_df[::200]
        # valid_df = train_df.copy()

    train_images = train_df[INPUT_IMAGE_ID_KEY].tolist()
    train_images = [os.path.join(data_dir, "Cover", x) for x in train_images]

    train_qf = train_df["quality"].values.tolist()

    # Validation
    valid_images = valid_df[INPUT_IMAGE_ID_KEY].tolist()
    valid_images = [os.path.join(data_dir, "Cover", x) for x in valid_images]

    valid_x = valid_images.copy()
    valid_y = [0] * len(valid_images)
    valid_qf = valid_df["quality"].values.tolist()

    for method_index, method_name in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
        # Filter images that does not have any alterations DCT (there are 250 of them)
        unchanged_files = unchanged[unchanged["method"] == method_name].file.values
        unchanged_files = list(map(fs.id_from_fname, unchanged_files))

        for fname, qf in zip(valid_images, valid_df["quality"].values):
            if fs.id_from_fname(fname) not in unchanged_files:
                fname = fname.replace("Cover", method_name)
                valid_x.append(fname)
                valid_y.append(method_index + 1)
                valid_qf.append(qf)
            else:
                # print("Removed unchanged file from the valid set", fname)
                pass

    train_ds = (
        PairedImageDataset(
            train_images, train_qf, target=1, transform=train_transform, features=features, bitmix=bitmix
        )
        + PairedImageDataset(
            train_images, train_qf, target=2, transform=train_transform, features=features, bitmix=bitmix
        )
        + PairedImageDataset(
            train_images, train_qf, target=3, transform=train_transform, features=features, bitmix=bitmix
        )
    )

    valid_ds = TrainingValidationDataset(
        images=valid_x, targets=valid_y, quality=valid_qf, transform=valid_transform, features=features
    )

    sampler = None
    print("Train", train_ds)
    print("Valid", valid_ds)
    return train_ds, valid_ds, sampler


def get_holdout(data_dir: str, features=None):
    valid_transform = A.NoOp()

    data_folds = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "folds_v2.csv"))
    unchanged = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "df_unchanged.csv"))

    changed_bits = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "changed_bits.csv"))
    changed_bits["key"] = (
        changed_bits["file"] + "_" + changed_bits["method"].apply(lambda x: METHOD_TO_INDEX[x]).apply(str)
    )
    changed_bits_table = {}
    for i, row in changed_bits.iterrows():
        changed_bits_table[row["key"]] = row["nbits"]

    # Take only holdout fold
    holdout_df = data_folds[data_folds[INPUT_FOLD_KEY] == HOLDOUT_FOLD]

    holdout_images = holdout_df[INPUT_IMAGE_ID_KEY].tolist()
    holdout_images = [os.path.join(data_dir, "Cover", x) for x in holdout_images]

    valid_x = holdout_images.copy()
    valid_y = [0] * len(holdout_images)
    valid_qf = holdout_df["quality"].values.tolist()
    valid_bits = [0] * len(holdout_images)

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

                key = os.path.basename(fname) + "_" + str(method_index)
                valid_bits.append(changed_bits_table[key])

            else:
                # print("Removed unchanged file from the holdout set", fname)
                pass

    holdout_ds = TrainingValidationDataset(
        images=valid_x,
        targets=valid_y,
        quality=valid_qf,
        bits=valid_bits,
        transform=valid_transform,
        features=features,
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
        bits=None,
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


def get_test_dataset(data_dir, features):
    valid_transform = A.NoOp()
    # images = fs.find_images_in_dir(os.path.join(data_dir, "Test"))
    test_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_dataset_qf_qt.csv"))
    test_df["image_fname"] = test_df["image_id"].apply(lambda x: os.path.join(data_dir, "Test", x))

    return TrainingValidationDataset(
        images=test_df["image_fname"].values.tolist(),
        targets=None,
        bits=None,
        quality=test_df["quality"].values.tolist(),
        transform=valid_transform,
        features=features,
    )


def get_istego100k_test_same(data_dir: str, features, output_size="full"):
    assert output_size in {"full", "random_crop", "center_crop", "tiles"}
    from .augmentations import RandomCrop8

    labels = json.load(open(os.path.join(data_dir, "same_source_test.parameter.json")))

    image_ids = []
    quality = []
    methods = []

    for image_id, kv in labels.items():
        image_ids.append(os.path.join(data_dir, "test_same", image_id))
        quality.append(int(kv["quality"]))
        methods.append(METHOD_TO_INDEX[kv.get("steg_algorithm", "Cover")])

    if output_size == "full":
        valid_transform = A.NoOp()
        return TrainingValidationDataset(
            images=image_ids, targets=methods, quality=quality, transform=valid_transform, features=features
        )
    elif output_size == "center_crop":
        valid_transform = A.CenterCrop(512, 512)
        return TrainingValidationDataset(
            images=image_ids, targets=methods, quality=quality, transform=valid_transform, features=features
        )
    elif output_size == "random_crop":
        valid_transform = RandomCrop8(512, 512)
        return TrainingValidationDataset(
            images=image_ids, targets=methods, quality=quality, transform=valid_transform, features=features
        )
    elif output_size == "tiles":
        return (
            TrainingValidationDataset(
                images=image_ids, targets=methods, quality=quality, transform=A.Crop(0, 0, 512, 512), features=features
            )
            + TrainingValidationDataset(
                images=image_ids,
                targets=methods,
                quality=quality,
                transform=A.Crop(512, 0, 1024, 512),
                features=features,
            )
            + TrainingValidationDataset(
                images=image_ids,
                targets=methods,
                quality=quality,
                transform=A.Crop(0, 512, 512, 1024),
                features=features,
            )
            + TrainingValidationDataset(
                images=image_ids,
                targets=methods,
                quality=quality,
                transform=A.Crop(512, 512, 1024, 1024),
                features=features,
            )
        )

    raise KeyError(output_size)


def get_istego100k_test_other(data_dir: str, features, output_size="full"):
    assert output_size in {"full", "random_crop", "center_crop", "tiles"}

    from .augmentations import RandomCrop8

    labels = json.load(open(os.path.join(data_dir, "different_source_test.parameter.json")))

    image_ids = []
    quality = []
    methods = []

    for image_id, kv in labels.items():
        image_ids.append(os.path.join(data_dir, "test_other", image_id))
        quality.append(int(kv["quality"]))
        methods.append(METHOD_TO_INDEX[kv.get("steg_algorithm", "Cover")])

    if output_size == "full":
        valid_transform = A.NoOp()
        return TrainingValidationDataset(
            images=image_ids, targets=methods, quality=quality, transform=valid_transform, features=features
        )
    elif output_size == "center_crop":
        valid_transform = A.CenterCrop(512, 512)
        return TrainingValidationDataset(
            images=image_ids, targets=methods, quality=quality, transform=valid_transform, features=features
        )
    elif output_size == "random_crop":
        valid_transform = RandomCrop8(512, 512)
        return TrainingValidationDataset(
            images=image_ids, targets=methods, quality=quality, transform=valid_transform, features=features
        )
    elif output_size == "tiles":
        return (
            TrainingValidationDataset(
                images=image_ids, targets=methods, quality=quality, transform=A.Crop(0, 0, 512, 512), features=features
            )
            + TrainingValidationDataset(
                images=image_ids,
                targets=methods,
                quality=quality,
                transform=A.Crop(512, 0, 1024, 512),
                features=features,
            )
            + TrainingValidationDataset(
                images=image_ids,
                targets=methods,
                quality=quality,
                transform=A.Crop(0, 512, 512, 1024),
                features=features,
            )
            + TrainingValidationDataset(
                images=image_ids,
                targets=methods,
                quality=quality,
                transform=A.Crop(512, 512, 1024, 1024),
                features=features,
            )
        )

    raise KeyError(output_size)


def get_istego100k_train(data_dir: str, fold: int, features, output_size="full"):
    assert output_size in {"full", "random_crop", "center_crop", "tiles"}
    from .augmentations import RandomCrop8

    labels = json.load(open(os.path.join(data_dir, "train.parameter.json")))

    image_ids = []
    qualities = []
    targets = []
    folds = []

    cover_images = set([os.path.basename(x) for x in fs.find_images_in_dir(os.path.join(data_dir, "train", "cover"))])
    stego_images = set([os.path.basename(x) for x in fs.find_images_in_dir(os.path.join(data_dir, "train", "stego"))])
    all_images = list(sorted(cover_images.union(stego_images)))

    for i, image_id in enumerate(all_images):
        fold_index = i % 4
        kv = labels.get(image_id, {})
        method = METHOD_TO_INDEX[kv.get("steg_algorithm", "Cover")]

        quality = int(kv.get("quality", 0))
        if quality not in {75, 90, 95}:
            continue

        if image_id in cover_images:
            image_ids.append(os.path.join(data_dir, "train", "cover", image_id))
            qualities.append(quality)
            targets.append(METHOD_TO_INDEX["Cover"])
            folds.append(fold_index)

        if image_id in stego_images and method in {1, 2, 3}:
            image_ids.append(os.path.join(data_dir, "train", "stego", image_id))
            qualities.append(quality)
            targets.append(method)
            folds.append(fold_index)

    image_ids = np.array(image_ids)
    qualities = np.array(qualities)
    targets = np.array(targets)
    folds = np.array(folds)

    image_ids = image_ids[folds != fold].tolist()
    qualities = qualities[folds != fold].tolist()
    targets = targets[folds != fold].tolist()

    print(len(all_images), len(image_ids))

    if output_size == "full":
        valid_transform = A.NoOp()
        train_ds = TrainingValidationDataset(
            images=image_ids, targets=targets, quality=qualities, transform=valid_transform, features=features
        )
        print("Extra dataset", train_ds)
    elif output_size == "center_crop":
        valid_transform = A.CenterCrop(512, 512)
        train_ds = TrainingValidationDataset(
            images=image_ids, targets=targets, quality=qualities, transform=valid_transform, features=features
        )
        print("Extra dataset", train_ds)
    elif output_size == "random_crop":
        valid_transform = RandomCrop8(512, 512)
        train_ds = TrainingValidationDataset(
            images=image_ids, targets=targets, quality=qualities, transform=valid_transform, features=features
        )
        print("Extra dataset", train_ds)
    elif output_size == "tiles":

        train_ds = [
            TrainingValidationDataset(
                images=image_ids,
                targets=targets,
                quality=qualities,
                transform=A.Crop(0, 0, 512, 512),
                features=features,
            ),
            TrainingValidationDataset(
                images=image_ids,
                targets=targets,
                quality=qualities,
                transform=A.Crop(512, 0, 1024, 512),
                features=features,
            ),
            TrainingValidationDataset(
                images=image_ids,
                targets=targets,
                quality=qualities,
                transform=A.Crop(0, 512, 512, 1024),
                features=features,
            ),
            TrainingValidationDataset(
                images=image_ids,
                targets=targets,
                quality=qualities,
                transform=A.Crop(512, 512, 1024, 1024),
                features=features,
            ),
        ]

        for ds in train_ds:
            print("Extra dataset", ds)
        train_ds = ConcatDataset(train_ds)

    return train_ds
