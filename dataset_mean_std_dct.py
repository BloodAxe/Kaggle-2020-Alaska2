import numpy as np
import cv2
import torch
from pytorch_toolbelt.utils import fs, to_numpy
from tqdm import tqdm

from alaska2 import idct8
from alaska2.dataset import idct8v2
from alaska2.models.dct import SpaceToDepth

sd2 = SpaceToDepth(block_size=8)


def compute_mean_std(dataset):
    """
    https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    """

    global_mean = np.zeros((3 * 64), dtype=np.float64)
    global_var = np.zeros((3 * 64), dtype=np.float64)

    n_items = 0

    for image_fname in dataset:
        dct_file = np.load(fs.change_extension(image_fname, ".npz"))
        y = torch.from_numpy(dct_file["dct_y"])
        cb = torch.from_numpy(dct_file["dct_cb"])
        cr = torch.from_numpy(dct_file["dct_cr"])

        dct = torch.stack([y, cb, cr], dim=0).unsqueeze(0)
        dct = to_numpy(sd2(dct)[0])

        global_mean += dct.mean(axis=(1, 2))
        global_var += dct.std(axis=(1, 2)) ** 2

        n_items += 1

    return global_mean / n_items, np.sqrt(global_var / n_items)


def main():
    dataset = fs.find_images_in_dir("d:\\datasets\\ALASKA2\\Cover")
    # dataset = dataset[:500]

    mean, std = compute_mean_std(tqdm(dataset))

    print("Y", mean, std)


if __name__ == "__main__":
    main()
