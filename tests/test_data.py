import pytest, os, cv2

from alaska2.dataset import *

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


def test_dct():
    image = cv2.imread(os.path.join(TEST_DATA_DIR, "Cover", "00001.jpg"), cv2.IMREAD_GRAYSCALE)
    dct = compute_dct(image)
    assert dct.shape[0] * 8 == image.shape[0]
    assert dct.shape[1] * 8 == image.shape[1]
    print(dct.shape, dct.mean(axis=(0, 1)), dct.std(axis=(0, 1)))
