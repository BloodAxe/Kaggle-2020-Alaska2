import os
from multiprocessing import Pool

import cv2
from pytorch_toolbelt.utils import fs
from sklearn.neighbors._ball_tree import BallTree
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


def to_np(x):
    y = np.fromstring(x[1:-1], sep=",", dtype=np.uint8)
    return y


def main():
    data_dir = "D:\datasets\ALASKA2"
    cover = fs.find_images_in_dir(os.path.join(data_dir, "Cover"))
    test = fs.find_images_in_dir(os.path.join(data_dir, "Test"))
    all_images = cover + test

    # all_images = all_images[:100]
    features = pd.read_csv("features.csv")

    t = np.stack(features["top"].apply(to_np))
    b = np.stack(features["bottom"].apply(to_np))
    l = np.stack(features["left"].apply(to_np))
    r = np.stack(features["right"].apply(to_np))

    radius = 0.05 * 255
    neigh = BallTree(t)
    num_samples = len(features)

    for query_index, b_i in enumerate(tqdm(b)):
        neigh_ind, neigh_dist = neigh.query_radius([b_i], radius, return_distance=True)

        for dist, ind in zip(neigh_dist, neigh_ind):
            if len(dist):
                print(dist, ind)
                train_ind = ind[0]
                query_image = cv2.imread(all_images[query_index])
                train_image = cv2.imread(all_images[train_ind])
                cv2.imshow("Query", query_image)
                cv2.imshow("Train", train_image)
                cv2.waitKey(-1)


if __name__ == "__main__":
    main()
