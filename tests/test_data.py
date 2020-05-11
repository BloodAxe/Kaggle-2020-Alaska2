import pytest, os, cv2
import numpy as np
import torch

from alaska2.dataset import *
import matplotlib.pyplot as plt


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


def test_compute_dct_slow():
    a = os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg")
    dct_y, dct_cr, dct_cb = compute_dct_slow(a)
    print(dct_y.shape, dct_y.mean(), dct_y.std())
    print(dct_cr.shape, dct_cr.mean(), dct_cr.std())
    print(dct_cb.shape, dct_cb.mean(), dct_cb.std())


def test_compute_dct_fast():
    a = os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg")
    dct_y, dct_cr, dct_cb = compute_dct_fast(a)
    print(dct_y.shape, dct_y.mean(), dct_y.std())
    print(dct_cr.shape, dct_cr.mean(), dct_cr.std())
    print(dct_cb.shape, dct_cb.mean(), dct_cb.std())


def test_interpolate2():
    a = os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg")
    i = cv2.imread(a)

    j = upsample2(i)
    cv2.imshow("i", i)
    cv2.imshow("j", j)
    cv2.waitKey(-1)


def test_dct():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"), cv2.IMREAD_GRAYSCALE)
    dct = compute_dct_fast(image)

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
