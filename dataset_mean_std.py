import numpy as np
import cv2
from pytorch_toolbelt.utils import fs
from tqdm import tqdm


def compute_mean_std(dataset, read_image=cv2.imread):
    """
    https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    """
    one_over_255 = float(1.0 / 255.0)

    global_mean = np.zeros(3, dtype=np.float64)
    global_var = np.zeros(3, dtype=np.float64)

    n_items = 0

    for image_fname in dataset:
        x = read_image(image_fname) * one_over_255
        mean, stddev = cv2.meanStdDev(x)

        global_mean += np.squeeze(mean)
        global_var += np.squeeze(stddev) ** 2
        n_items += 1

    return global_mean / n_items, np.sqrt(global_var / n_items)


dataset = fs.find_images_in_dir("d:\\datasets\\ALASKA2\\Cover")
print(compute_mean_std(tqdm(dataset)))
