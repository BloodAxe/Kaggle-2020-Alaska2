import cv2
import numpy as np
from alaska2.dataset import compute_decoding_residual


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

    cover = cv2.imread("d:/datasets/ALASKA2/Cover/38330.jpg")
    stego = cv2.imread("d:/datasets/ALASKA2/UERD/38330.jpg")
    r1 = compute_decoding_residual(cover, "d:/datasets/ALASKA2/Cover/38330.npz")
    r2 = compute_decoding_residual(stego, "d:/datasets/ALASKA2/UERD/38330.npz")

    # cover = cv2.imread("d:/datasets/ALASKA2/Cover/20805.jpg")
    # stego = cv2.imread("d:/datasets/ALASKA2/UERD/20805.jpg")
    # r1 = compute_decoding_residual(cover, "d:/datasets/ALASKA2/Cover/20805.npz")
    # r2 = compute_decoding_residual(stego, "d:/datasets/ALASKA2/UERD/20805.npz")


    r1b = block8_sum(np.abs(r1).sum(axis=2))
    r2b = block8_sum(np.abs(r2).sum(axis=2))

    diff = (r1 - r2).sum(axis=2)

    print(diff)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(cover)
    plt.show()

    plt.figure()
    plt.imshow(diff)
    plt.show()

    plt.figure()
    plt.imshow(cv2.absdiff(cover, stego).sum(axis=2), cmap="jet")
    plt.show()

    plt.figure()
    plt.imshow(block8_sum(cv2.absdiff(cover, stego).sum(axis=2)), cmap="jet")
    plt.show()

    plt.figure()
    plt.imshow(block8_sum(cv2.absdiff(cover, stego).sum(axis=2)) > 0, cmap="jet")
    plt.show()

    plt.figure()
    plt.imshow(r1b)
    plt.show()

    plt.figure()
    plt.imshow(r2b)
    plt.show()
