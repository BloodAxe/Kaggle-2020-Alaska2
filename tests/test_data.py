import pytest, os, cv2
import numpy as np
import torch

from alaska2.dataset import *
import matplotlib.pyplot as plt

from alaska2.dataset import image_dct_slow
from alaska2.models.dct import DCT

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

import jpeg4py as jpeg

def test_dct():
    a = os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg")
    d = jpeg.JPEG(a)


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

    dct_tensor = dct_tensor[0,0].detach().cpu().numpy()
    print(dct_tensor.shape, dct2.mean(axis=(0, 1)), dct2.std(axis=(0, 1)))


def test_rgb_dct():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"))
    dct = compute_rgb_dct(image)
    assert dct.shape[0] * 8 == image.shape[0]
    assert dct.shape[1] * 8 == image.shape[1]
    print(dct.shape, dct.mean(axis=(0, 1)), dct.std(axis=(0, 1)))


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
    ela = np.abs(ela)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JMiPOD", "00001.jpg"), cv2.IMREAD_COLOR)
    ela = compute_ela(image)
    ela = np.abs(ela)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "JUNIWARD", "00001.jpg"), cv2.IMREAD_COLOR)
    ela = compute_ela(image)
    ela = np.abs(ela)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))

    image = cv2.imread(os.path.join(TEST_DATA_DIR, "UERD", "00001.jpg"), cv2.IMREAD_COLOR)
    ela = compute_ela(image)
    ela = np.abs(ela)

    print(ela.shape, ela.sum(axis=(0, 1)), ela.mean(axis=(0, 1)), ela.std(axis=(0, 1)))
