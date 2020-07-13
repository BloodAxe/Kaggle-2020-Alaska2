import cv2
import numpy as np
from alaska2.dataset import compute_decoding_residual, decode_bgr_from_dct


def block8_sum(a: np.ndarray):
    shape = a.shape
    a = a.reshape([a.shape[0] // 8, 8, a.shape[1] // 8, 8] + list(a.shape[2:]))
    a = a.sum(axis=(1, 3))
    return a


def test_decoding_residual():
    # d:\datasets\ALASKA2\Cover\20805.jpg
    # d:\datasets\ALASKA2\UERD\20805.jpg

    # /home/bloodaxe/datasets/ALASKA2/Cover/18051.jpg
    # /home/bloodaxe/datasets/ALASKA2/UERD/18051.jpg

    # cover = cv2.imread("d:/datasets/ALASKA2/Cover/38330.jpg")
    # stego = cv2.imread("d:/datasets/ALASKA2/UERD/38330.jpg")

    # r1 = compute_decoding_residual(cover, "d:/datasets/ALASKA2/Cover/38330.npz")
    # r2 = compute_decoding_residual(stego, "d:/datasets/ALASKA2/UERD/38330.npz")

    cover = 255 * decode_bgr_from_dct(
        "d:/datasets/ALASKA2/Cover/50846.npz"
    )  # cv2.imread("d:/datasets/ALASKA2/Cover/50846.jpg")
    cover_cv2 = cv2.imread("d:/datasets/ALASKA2/Cover/50846.jpg")

    cv2.imwrite("cover_read_with_cv2.png", cover_cv2)
    cv2.imwrite("cover_from_dct.png", cover.astype(np.uint8))

    stego = 255 * decode_bgr_from_dct("d:/datasets/ALASKA2/JMiPOD/50846.npz")

    print(cover.max(), cover.min())
    cv2.imshow("Cover", cover.astype(np.uint8))
    cv2.imshow("Stego", stego.astype(np.uint8))
    cv2.waitKey(-1)
    # cover = cv2.imread("d:/datasets/ALASKA2/Cover/20805.jpg")
    # stego = cv2.imread("d:/datasets/ALASKA2/UERD/20805.jpg")
    # r1 = compute_decoding_residual(cover, "d:/datasets/ALASKA2/Cover/20805.npz")
    # r2 = compute_decoding_residual(stego, "d:/datasets/ALASKA2/UERD/20805.npz")
