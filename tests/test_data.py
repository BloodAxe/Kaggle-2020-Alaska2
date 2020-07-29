import random
from collections import defaultdict

import pytest, os, cv2
import numpy as np
import torch
from pytorch_toolbelt.utils import rgb_image_from_tensor

from alaska2.augmentations import (
    dct_rot90_block,
    dct_rot90,
    dct_transpose,
    dct_transpose_block,
    dct_rot90_fast,
    dct_transpose_fast,
)
from alaska2.dataset import *
import matplotlib.pyplot as plt

from alaska2.dataset import (
    compute_dct_fast,
    compute_dct_slow,
    idct8v2,
    dct2spatial,
    dct2channels_last,
    decode_bgr_from_dct,
    compute_ela_rich,
    compute_decoding_residual,
)

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


def test_batched_ds():
    train_ds, valid_ds, _ = get_datasets_paired(
        data_dir=os.environ.get("KAGGLE_2020_ALASKA2"), fold=0, features=[INPUT_IMAGE_KEY]
    )
    sample = train_ds[0]
    sample = train_ds[len(train_ds) - 1]


import matplotlib.pyplot as plt


def test_dct_rot90():
    image_fname = os.path.join(TEST_DATA_DIR, "Cover", "00002.jpg")
    image_0 = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE) / 255.0

    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(image_0, cmap="gray")
    # ax[1].imshow(idct8(dct2channels_last(dct2spatial(dct8(image_0)))), cmap="gray")
    # plt.show()

    # dct_cv = dct2spatial(dct8(image_0))
    # dct_cv = np.dstack([dct_cv, dct_cv, dct_cv])
    dct_cv = compute_features(None, image_fname, [INPUT_FEATURES_DCT_KEY])[INPUT_FEATURES_DCT_KEY]

    image_0 = image_0[256 : 256 + 8, 256 : 256 + 8]
    image_1 = np.rot90(image_0, 1)
    image_2 = np.rot90(image_0, 2)
    image_3 = np.rot90(image_0, 3)

    expected_dct88_0 = cv2.dct(image_0)
    expected_dct88_1 = cv2.dct(image_1)
    expected_dct88_2 = cv2.dct(image_2)
    expected_dct88_3 = cv2.dct(image_3)

    # actual_dct88_0 = dct_rot90_block(expected_dct88_0, 0)
    # actual_dct88_1 = dct_rot90_block(expected_dct88_0, 1)
    # actual_dct88_2 = dct_rot90_block(expected_dct88_0, 2)
    # actual_dct88_3 = dct_rot90_block(expected_dct88_0, 3)
    # f, ax = plt.subplots(3, 4)
    # ax[0, 0].imshow(image_0)
    # ax[0, 1].imshow(image_1)
    # ax[0, 2].imshow(image_2)
    # ax[0, 3].imshow(image_3)
    # ax[1, 0].imshow(cv2.idct(expected_dct88_0))
    # ax[1, 1].imshow(cv2.idct(expected_dct88_1))
    # ax[1, 2].imshow(cv2.idct(expected_dct88_2))
    # ax[1, 3].imshow(cv2.idct(expected_dct88_3))
    #
    # ax[2, 0].imshow(cv2.idct(actual_dct88_0))
    # ax[2, 1].imshow(cv2.idct(actual_dct88_1))
    # ax[2, 2].imshow(cv2.idct(actual_dct88_2))
    # ax[2, 3].imshow(cv2.idct(actual_dct88_3))
    # f.show()

    # dct_000 = dct_rot90(dct, 1)
    dct_1 = dct_rot90(dct_cv, 0)
    dct_2 = dct_rot90(dct_cv, 1)
    dct_3 = dct_rot90(dct_cv, 2)
    dct_4 = dct_rot90(dct_cv, 3)

    f, ax = plt.subplots(2, 2, figsize=(16, 16))
    ax[0, 0].imshow(idct8(dct2channels_last(dct_1[..., 0])), cmap="gray")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(idct8(dct2channels_last(dct_2[..., 0])), cmap="gray")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(idct8(dct2channels_last(dct_3[..., 0])), cmap="gray")
    ax[1, 0].axis("off")
    ax[1, 1].imshow(idct8(dct2channels_last(dct_4[..., 0])), cmap="gray")
    ax[1, 1].axis("off")
    f.show()

    dct_1 = dct_rot90_fast(dct_cv, 0)
    dct_2 = dct_rot90_fast(dct_cv, 1)
    dct_3 = dct_rot90_fast(dct_cv, 2)
    dct_4 = dct_rot90_fast(dct_cv, 3)

    f, ax = plt.subplots(2, 2, figsize=(16, 16))
    ax[0, 0].imshow(idct8(dct2channels_last(dct_1[..., 0])), cmap="gray")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(idct8(dct2channels_last(dct_2[..., 0])), cmap="gray")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(idct8(dct2channels_last(dct_3[..., 0])), cmap="gray")
    ax[1, 0].axis("off")
    ax[1, 1].imshow(idct8(dct2channels_last(dct_4[..., 0])), cmap="gray")
    ax[1, 1].axis("off")
    f.show()


def test_dct_transpose():
    image_fname = os.path.join(TEST_DATA_DIR, "Cover", "00002.jpg")
    image_0 = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE) / 255.0

    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(image_0, cmap="gray")
    # ax[1].imshow(idct8(dct2channels_last(dct2spatial(dct8(image_0)))), cmap="gray")
    # plt.show()

    dct_cv = dct2spatial(dct8(image_0))
    dct_cv = np.dstack([dct_cv, dct_cv, dct_cv])
    # dct_cv = compute_features(None, image_fname, [INPUT_FEATURES_DCT_KEY])[INPUT_FEATURES_DCT_KEY]

    image_0 = image_0[256 : 256 + 8, 256 : 256 + 8]
    image_1 = np.transpose(image_0)

    expected_dct88_0 = cv2.dct(image_0)
    expected_dct88_1 = cv2.dct(image_1)

    actual_dct88_1 = dct_transpose_block(expected_dct88_0)
    f, ax = plt.subplots(3, 2)
    ax[0, 0].imshow(image_0)
    ax[0, 1].imshow(image_1)

    ax[1, 0].imshow(cv2.idct(expected_dct88_0))
    ax[1, 1].imshow(cv2.idct(expected_dct88_1))

    ax[2, 0].imshow(cv2.idct(expected_dct88_0))
    ax[2, 1].imshow(cv2.idct(actual_dct88_1))
    f.show()

    # dct_000 = dct_rot90(dct, 1)
    dct_1 = dct_cv
    dct_2 = dct_transpose(dct_cv)

    f, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(idct8(dct2channels_last(dct_1[..., 0])), cmap="gray")
    ax[0].axis("off")
    ax[1].imshow(idct8(dct2channels_last(dct_2[..., 0])), cmap="gray")
    ax[1].axis("off")
    f.show()

    dct_1 = dct_cv
    dct_2 = dct_transpose_fast(dct_cv)

    f, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(idct8(dct2channels_last(dct_1[..., 0])), cmap="gray")
    ax[0].axis("off")
    ax[1].imshow(idct8(dct2channels_last(dct_2[..., 0])), cmap="gray")
    ax[1].axis("off")
    f.show()


def test_dct():
    import jpegio as jpio

    image_fname = os.path.join(TEST_DATA_DIR, "Cover", "00002.jpg")
    image = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE)
    dct_y, dct_cb, dct_cr = compute_dct_fast(image_fname)
    y1 = idct8(dct_y)
    cb1 = idct8(dct_cb)
    cr1 = idct8(dct_cr)

    y88 = y1[:8, :8, 0]
    dct_y88 = dct_y[0, 0].reshape((8, 8))

    jpegStruct = jpio.read(image_fname)
    qt = jpegStruct.quant_tables
    dct_matrix = jpegStruct.coef_arrays

    dct_y88_2 = jpegStruct.coef_arrays[0][:8, :8] * qt[0]
    y2 = idct8v2(jpegStruct.coef_arrays[0], qt[0])

    y2_88 = y2[:8, :8, 0]

    cb2 = idct8v2(jpegStruct.coef_arrays[1], qt[1])
    cr2 = idct8v2(jpegStruct.coef_arrays[2], qt[1])
    print(dct_matrix)


def test_dct_comp():

    methods = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]

    cover_diff = None

    f, ax = plt.subplots(4, 4, figsize=(8 * 4, 8 * 4))
    for i, method in enumerate(methods):
        bgr = cv2.imread(os.path.join(TEST_DATA_DIR, method, "00001.jpg"))

        bgr_from_dct = decode_bgr_from_dct(os.path.join(TEST_DATA_DIR, method, "00001.npz"))
        bgr_from_dct_byte = (bgr_from_dct * 255).astype(np.uint8)

        if i == 0:
            cover_diff = cv2.absdiff(bgr_from_dct * 255, bgr.astype(np.float32)).sum(axis=2)

        decode_diff = cv2.absdiff(bgr_from_dct * 255, bgr.astype(np.float32)).sum(axis=2)
        print(decode_diff.mean(), decode_diff.std())
        ax[i, 0].imshow(bgr)
        ax[i, 0].axis("off")
        ax[i, 1].imshow(bgr_from_dct_byte)
        ax[i, 1].axis("off")
        ax[i, 2].imshow(decode_diff, cmap="gray")
        ax[i, 2].axis("off")
        ax[i, 3].imshow(cv2.absdiff(decode_diff, cover_diff), cmap="gray")
        ax[i, 3].axis("off")
    plt.show()


def test_blur_features():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    feature_maps1 = compute_blur_features(image)

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.jpg"))
    feature_maps2 = compute_blur_features(image)

    plt.figure()
    for fm in range(feature_maps1.shape[2]):
        plt.matshow(feature_maps1[..., fm] - feature_maps2[..., fm], fm)
    plt.show()


def test_bitmix():
    cover = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    stego = cv2.imread(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.jpg"))

    c, s, lc, ls, m = bitmix(cover, stego, 0.125)
    print(lc, ls)

    cv2.imshow("Cover", c)
    cv2.imshow("Stego", s)
    cv2.imshow("Mask", m * 255)
    cv2.waitKey(-1)


def test_ela():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    ela = compute_ela_rich(image)
    mean, std = ela.mean(axis=(0, 1)), ela.std(axis=(0, 1))
    print(ela.shape, ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.jpg"))
    ela1 = (compute_ela_rich(image) - mean) / std
    print(ela1.shape, ela1.mean(axis=(0, 1)), ela1.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JUNIWARD", "00001.jpg"))
    ela2 = (compute_ela_rich(image) - mean) / std
    print(ela2.shape, ela2.mean(axis=(0, 1)), ela2.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "UERD", "00001.jpg"))
    ela3 = (compute_ela_rich(image) - mean) / std
    print(ela3.shape, ela3.mean(axis=(0, 1)), ela3.std(axis=(0, 1)))


def test_normalize_ycrcb():
    files = [
        "../test_data/Cover/00001.npz",
        "../test_data/Cover/00002.npz",
        "../test_data/Cover/00003.npz",
        "../test_data/Cover/00004.npz",
    ]

    sample = defaultdict(list)

    for f in files:
        dct_file = np.load(f)
        sample[INPUT_FEATURES_CHANNEL_Y_KEY].append(idct8(dct_file["dct_y"]) / 64)
        sample[INPUT_FEATURES_CHANNEL_CR_KEY].append(idct8(dct_file["dct_cr"]) / 8)
        sample[INPUT_FEATURES_CHANNEL_CB_KEY].append(idct8(dct_file["dct_cb"]) / 8)

    y = np.row_stack(sample[INPUT_FEATURES_CHANNEL_Y_KEY])
    cr = np.row_stack(sample[INPUT_FEATURES_CHANNEL_CR_KEY])
    cb = np.row_stack(sample[INPUT_FEATURES_CHANNEL_CB_KEY])

    print("Y ", y.mean(), y.std())
    print("Cr", cr.mean(), cr.std())
    print("Cb", cb.mean(), cb.std())


def test_randint():
    import matplotlib.pyplot as plt

    samples = []
    for _ in range(10000):
        samples.append(random.randint(0, 2))

    plt.figure()
    plt.hist(samples)
    plt.show()


def test_paired_ds():
    train_ds, _, _ = get_datasets_paired("d://datasets//ALASKA2", 0, features=[INPUT_IMAGE_KEY])

    for i in range(100):
        sample = train_ds[i]
        assert sample[INPUT_TRUE_MODIFICATION_FLAG][0] == 0
        assert sample[INPUT_TRUE_MODIFICATION_FLAG][1] == 1
        assert sample[INPUT_TRUE_MODIFICATION_TYPE][0] == 0
        assert sample[INPUT_TRUE_MODIFICATION_TYPE][1] > 0

        cv2.imshow("Cover", rgb_image_from_tensor(sample[INPUT_IMAGE_KEY][0], mean=0.0, std=1.0, max_pixel_value=1))
        cv2.imshow("Stego", rgb_image_from_tensor(sample[INPUT_IMAGE_KEY][1], mean=0.0, std=1.0, max_pixel_value=1))
        cv2.imshow(
            "Diff",
            40
            * cv2.absdiff(
                rgb_image_from_tensor(sample[INPUT_IMAGE_KEY][0], mean=0.0, std=1.0, max_pixel_value=1),
                rgb_image_from_tensor(sample[INPUT_IMAGE_KEY][1], mean=0.0, std=1.0, max_pixel_value=1),
            ),
        )

        cv2.waitKey(-1)
