import numpy as np
import cv2
from pytorch_toolbelt.utils import fs
from tqdm import tqdm

from alaska2 import idct8
from alaska2.dataset import idct8v2


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
        y = idct8v2(dct_file["dct_y"], dct_file["quant_table"][0])
        cb = idct8v2(dct_file["dct_cb"], dct_file["quant_table"][1])
        cr = idct8v2(dct_file["dct_cr"], dct_file["quant_table"][1])

        global_mean[0] += y.mean()
        global_mean[1] += cb.mean()
        global_mean[2] += cr.mean()

        global_var[0] += y.std() ** 2
        global_var[1] += cb.std() ** 2
        global_var[2] += cr.std() ** 2

        n_items += 1

    return global_mean / n_items, np.sqrt(global_var / n_items)


def main():
    dataset = fs.find_images_in_dir("/home/bloodaxe/datasets/ALASKA2/Cover")
    dataset = dataset[:500]
    print("YCbCr", compute_mean_std(tqdm(dataset)))


if __name__ == "__main__":
    main()
