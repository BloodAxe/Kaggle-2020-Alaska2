import pytest, os, cv2
import numpy as np
from alaska2.dataset import *
import matplotlib.pyplot as plt

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


def test_dct():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"), cv2.IMREAD_GRAYSCALE)
    dct = compute_dct(image)
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
