from collections import defaultdict

import pytest, os, cv2
import numpy as np
import torch

from alaska2.dataset import *
import matplotlib.pyplot as plt

from alaska2.dataset import compute_dct_fast, compute_dct_slow

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


def test_dct():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"), cv2.IMREAD_GRAYSCALE)
    dct = compute_dct(image)

    assert dct.shape[0] * 8 == image.shape[0]
    assert dct.shape[1] * 8 == image.shape[1]
    print(dct.shape, dct.mean(axis=(0, 1)), dct.std(axis=(0, 1)))

    one_over_255 = np.float32(1.0 / 255.0)
    image = image * one_over_255

    dct2 = image_dct_slow(image)
    print(dct2.shape, dct2.mean(axis=(0, 1)), dct2.std(axis=(0, 1)))
    # np.testing.assert_allclose(dct2, dct)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    net = DCT()
    dct_tensor = net(image_tensor)

    dct_tensor = dct_tensor[0, 0].detach().cpu().numpy()
    print(dct_tensor.shape, dct2.mean(axis=(0, 1)), dct2.std(axis=(0, 1)))


def test_dct_comp():
    dct1 = np.load(os.path.join(TEST_DATA_DIR, "Cover", "00001.npz"))
    dct2 = np.load(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.npz"))
    y1 = dct1["dct_y"]
    y2 = dct2["dct_y"]
    print(dct1)


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
