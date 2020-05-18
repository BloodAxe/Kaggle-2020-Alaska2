import numpy as np
import cv2
from pytorch_toolbelt.utils import fs
from tqdm import tqdm

from alaska2 import idct8


def compute_mean_std(dataset):
    """
    https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    """

    global_mean = np.zeros(3, dtype=np.float64)
    global_var = np.zeros(3, dtype=np.float64)

    n_items = 0

    for image_fname in dataset:
        dct_file = np.load(fs.change_extension(image_fname, ".npz"))
        # This normalization roughly puts values into zero mean and unit variance
        y = idct8(dct_file["dct_y"])
        cr = idct8(dct_file["dct_cr"])
        cb = idct8(dct_file["dct_cb"])

        global_mean[0] += y.mean()
        global_mean[1] += cr.mean()
        global_mean[2] += cb.mean()

        global_var[0] += y.std() ** 2
        global_var[1] += cr.std() ** 2
        global_var[2] += cb.std() ** 2

        n_items += 1

    return global_mean / n_items, np.sqrt(global_var / n_items)


def main():
    dataset = fs.find_images_in_dir("d:\\datasets\\ALASKA2\\Cover")
    print("YCrCB", compute_mean_std(tqdm(dataset)))


if __name__ == "__main__":
    main()
