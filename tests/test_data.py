import random
from collections import defaultdict

import pytest, os, cv2
import numpy as np
import torch

from alaska2.dataset import *
import matplotlib.pyplot as plt

from alaska2.dataset import compute_dct_fast, compute_dct_slow, idct8v2

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


def test_batched_ds():
    train_ds, valid_ds, _ = get_datasets_paired(
        data_dir=os.environ.get("KAGGLE_2020_ALASKA2"), fold=0, features=[INPUT_IMAGE_KEY]
    )
    sample = train_ds[0]
    sample = train_ds[len(train_ds) - 1]


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
    [Col, Row] = np.meshgrid(range(8), range(8))
    T = 0.5 * np.cos(np.pi * (2 * Col + 1) * Row / (2 * 8))
    T[0, :] = T[0, :] / np.sqrt(2)

    print(T)

    dct1 = np.load(os.path.join(TEST_DATA_DIR, "Cover", "00001.npz"))

    y = idct8(dct1["dct_y"]).squeeze(-1)
    cr = idct8(dct1["dct_cr"]).squeeze(-1)
    cb = idct8(dct1["dct_cb"]).squeeze(-1)

    plt.figure()
    plt.imshow(y, cmap="gray")
    plt.show()

    plt.figure()
    plt.imshow(cr)
    plt.show()

    plt.figure()
    plt.imshow(cb)
    plt.show()

    imgYCC = np.dstack(
        [
            y / 128.0,
            cv2.resize(cr / 64.0, None, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
            cv2.resize(cb / 64.0, None, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
        ]
    )
    rgb = cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2BGR)

    plt.figure()
    plt.imshow(rgb)
    plt.show()

    # dct2 = np.load(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.npz"))
    # y1 = dct1["dct_y"]
    # y2 = dct2["dct_y"]
    # print(dct1)


def test_blur_features():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    feature_maps1 = compute_blur_features(image)

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.jpg"))
    feature_maps2 = compute_blur_features(image)

    plt.figure()
    for fm in range(feature_maps1.shape[2]):
        plt.matshow(feature_maps1[..., fm] - feature_maps2[..., fm], fm)
    plt.show()


def test_ela():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"), cv2.IMREAD_COLOR)
    ela = compute_ela(image)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.jpg"), cv2.IMREAD_COLOR)
    ela = compute_ela(image)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JUNIWARD", "00001.jpg"), cv2.IMREAD_COLOR)
    ela = compute_ela(image)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "UERD", "00001.jpg"), cv2.IMREAD_COLOR)
    ela = compute_ela(image)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))


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
