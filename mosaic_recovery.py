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
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import binary_fill_holes
import cv2  # To read and manipulate images
import os  # For filepath, directory handling
import sys  # System-specific parameters and functions
import tqdm  # Use smart progress meter
import seaborn as sns  # For pairplots
import matplotlib.pyplot as plt  # Python 2D plotting library
import matplotlib.cm as cm  # Color map

from sklearn.neighbors import NearestNeighbors

# Collection of methods for data operations. Implemented are functions to read
# images/masks from files and to read basic properties of the train/test data sets.


def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None, space="bgr"):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if space == "hsv":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(test_dir))[1]):
        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(
            ["{}".format(img_name_id), img_shape[0], img_shape[1], img_shape[0] / img_shape[1], img_shape[2], img_path]
        )

    test_df = pd.DataFrame(
        tmp, columns=["img_id", "img_height", "img_width", "img_ratio", "num_channels", "image_path"]
    )
    return test_df


def get_domimant_colors(img, top_colors=1):
    """Return dominant image color"""
    img = cv2.imread(img)
    img = cv2.pyrDown(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters=top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist


def cluster_images_by_hsv(images):
    """Clusterization based on hsv colors. Adds 'hsv_cluster' column to tables"""
    print("Loading data")
    # x_train_hsv, x_test_hsv = load_raw_data(image_size=None, space='hsv', load_mask=False)
    # x_hsv = np.concatenate([x_train_hsv, x_test_hsv])
    print("Calculating dominant hsv for each image")
    dominant_hsv = []
    for img in tqdm.tqdm(images):
        res1, res2 = get_domimant_colors(img, top_colors=1)
        dominant_hsv.append(res1.squeeze())
    print("Calculating clusters")
    kmeans = KMeans(n_clusters=3).fit(dominant_hsv)
    print("Images clustered")
    return kmeans.predict(dominant_hsv)


def plot_images(selected_images_df, images_rows=4, images_cols=8, plot_figsize=4):
    """Plot image_rows*image_cols of selected images. Used to visualy check clusterization"""
    f, axarr = plt.subplots(images_rows, images_cols, figsize=(plot_figsize * images_cols, images_rows * plot_figsize))
    for row in range(images_rows):
        for col in range(images_cols):
            if (row * images_cols + col) < selected_images_df.shape[0]:
                image_path = selected_images_df["image_path"].iloc[row * images_cols + col]
            else:
                continue
            img = read_image(image_path)
            height, width, l = img.shape
            ax = axarr[row, col]
            ax.axis("off")
            ax.set_title("%dx%d" % (width, height))
            ax.imshow(img)


def combine_images(data, indexes):
    """ Combines img from data using indexes as follows:
        0 1
        2 3
    """
    up = np.hstack([cv2.imread(data[indexes[0]]), cv2.imread(data[indexes[1]])])
    down = np.hstack([cv2.imread(data[indexes[2]]), cv2.imread(data[indexes[3]])])
    full = np.vstack([up, down])
    return full


def make_mosaic(image_fnames, return_connectivity=False, plot_images=False):
    """Find images with simular borders and combine them to one big image"""

    # extract borders from images
    borders = []
    for x in tqdm(image_fnames):
        x = cv2.imread(x)
        borders.extend([x[0, :, :].flatten(), x[-1, :, :].flatten(), x[:, 0, :].flatten(), x[:, -1, :].flatten()])
    borders = np.array(borders)

    # prepare df with all data
    lens = np.array([len(border) for border in borders])
    img_idx = list(range(len(image_fnames))) * 4
    img_idx.sort()
    position = ["up", "down", "left", "right"] * len(image_fnames)
    nn = [None] * len(position)
    df = pd.DataFrame(
        data=np.vstack([img_idx, position, borders, lens, nn]).T,
        columns=["img_idx", "position", "border", "len", "nn"],
    )
    uniq_lens = df["len"].unique()

    for idx, l in enumerate(uniq_lens):
        # fit NN on borders of certain size with 1 neighbor
        nn = NearestNeighbors(n_neighbors=1).fit(np.stack(df[df.len == l]["border"].values))
        distances, neighbors = nn.kneighbors()
        real_neighbor = np.array([None] * len(neighbors))
        distances, neighbors = distances.flatten(), neighbors.flatten()

        # if many borders are close to one, we want to take only the closest
        uniq_neighbors = np.unique(neighbors)

        # difficult to understand but works :c
        for un_n in uniq_neighbors:
            # min distance for borders with same nn
            min_index = list(distances).index(distances[neighbors == un_n].min())
            # check that min is double-sided
            double_sided = distances[neighbors[min_index]] == distances[neighbors == un_n].min()
            if double_sided and distances[neighbors[min_index]] < 1000:
                real_neighbor[min_index] = neighbors[min_index]
                real_neighbor[neighbors[min_index]] = min_index
        indexes = df[df.len == l].index
        for idx2, r_n in enumerate(real_neighbor):
            if r_n is not None:
                df["nn"].iloc[indexes[idx2]] = indexes[r_n]

    # img connectivity graph.
    img_connectivity = {}
    for img in df.img_idx.unique():
        slc = df[df["img_idx"] == img]
        img_nn = {}

        # get near images_id & position
        for nn_border, position in zip(slc[slc["nn"].notnull()]["nn"], slc[slc["nn"].notnull()]["position"]):

            # filter obvious errors when we try to connect bottom of one image to bottom of another
            # my hypotesis is that images were simply cut, without rotation
            if position == df.iloc[nn_border]["position"]:
                continue
            img_nn[position] = df.iloc[nn_border]["img_idx"]
        img_connectivity[img] = img_nn

    imgs = []
    indexes = set()
    mosaic_idx = 0

    # errors in connectivity are filtered
    good_img_connectivity = {}
    for k, v in img_connectivity.items():
        if v.get("down") is not None:
            if v.get("right") is not None:
                # need down right image
                # check if both right and down image are connected to the same image in the down right corner
                if (img_connectivity[v["right"]].get("down") is not None) and img_connectivity[v["down"]].get(
                    "right"
                ) is not None:
                    if img_connectivity[v["right"]]["down"] == img_connectivity[v["down"]]["right"]:
                        v["down_right"] = img_connectivity[v["right"]]["down"]
                        temp_indexes = [k, v["right"], v["down"], v["down_right"]]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        # надо тут фильтровать что они не одинаковые
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(image_fnames, temp_indexes))

                        # if external_df is not None:
                        #     external_df["mosaic_idx"].iloc[temp_indexes] = mosaic_idx
                        #     external_df["mosaic_position"].iloc[temp_indexes] = [
                        #         "up_left",
                        #         "up_right",
                        #         "down_left",
                        #         "down_right",
                        #     ]
                        #     mosaic_idx += 1
                        continue
            if v.get("left") is not None:
                # need down left image
                if (
                    img_connectivity[v["left"]].get("down") is not None
                    and img_connectivity[v["down"]].get("left") is not None
                ):
                    if img_connectivity[v["left"]]["down"] == img_connectivity[v["down"]]["left"]:
                        v["down_left"] = img_connectivity[v["left"]]["down"]
                        temp_indexes = [v["left"], k, v["down_left"], v["down"]]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(image_fnames, temp_indexes))

                        # if external_df is not None:
                        #     external_df["mosaic_idx"].iloc[temp_indexes] = mosaic_idx
                        #     external_df["mosaic_position"].iloc[temp_indexes] = [
                        #         "up_left",
                        #         "up_right",
                        #         "down_left",
                        #         "down_right",
                        #     ]
                        #
                        #     mosaic_idx += 1
                        continue
        if v.get("up") is not None:
            if v.get("right") is not None:
                # need up right image
                if (
                    img_connectivity[v["right"]].get("up") is not None
                    and img_connectivity[v["up"]].get("right") is not None
                ):
                    if img_connectivity[v["right"]]["up"] == img_connectivity[v["up"]]["right"]:
                        v["up_right"] = img_connectivity[v["right"]]["up"]
                        temp_indexes = [v["up"], v["up_right"], k, v["right"]]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(image_fnames, temp_indexes))

                        # if external_df is not None:
                        #     external_df["mosaic_idx"].iloc[temp_indexes] = mosaic_idx
                        #     external_df["mosaic_position"].iloc[temp_indexes] = [
                        #         "up_left",
                        #         "up_right",
                        #         "down_left",
                        #         "down_right",
                        #     ]
                        #
                        #     mosaic_idx += 1
                        continue
            if v.get("left") is not None:
                # need up left image
                if (
                    img_connectivity[v["left"]].get("up") is not None
                    and img_connectivity[v["up"]].get("left") is not None
                ):
                    if img_connectivity[v["left"]]["up"] == img_connectivity[v["up"]]["left"]:
                        v["up_left"] = img_connectivity[v["left"]]["up"]
                        temp_indexes = [v["up_left"], v["up"], v["left"], k]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(image_fnames, temp_indexes))

                        # if external_df is not None:
                        #     external_df["mosaic_idx"].iloc[temp_indexes] = mosaic_idx
                        #     external_df["mosaic_position"].iloc[temp_indexes] = [
                        #         "up_left",
                        #         "up_right",
                        #         "down_left",
                        #         "down_right",
                        #     ]
                        #
                        #     mosaic_idx += 1
                        continue

    # same images are present 4 times (one for every piece) so we need to filter them
    print("Images before filtering: {}".format(np.shape(imgs)))

    # can use np. unique only on images of one size, flatten first, then select
    flattened = np.array([i.flatten() for i in imgs])
    uniq_lens = np.unique([i.shape for i in flattened])
    filtered_imgs = []
    for un_l in uniq_lens:
        filtered_imgs.extend(np.unique(np.array([i for i in imgs if i.flatten().shape == un_l]), axis=0))

    filtered_imgs = np.array(filtered_imgs)
    print("Images after filtering: {}".format(np.shape(filtered_imgs)))

    if return_connectivity:
        print(good_img_connectivity)

    if plot_images:
        for i in filtered_imgs:
            plt.imshow(i)
            plt.show()

    # list of not combined images. return if you need
    not_combined = list(set(range(len(image_fnames))) - indexes)

    if return_connectivity:
        return filtered_imgs, good_img_connectivity
    else:
        return filtered_imgs


def main():
    data_dir = "D:\datasets\ALASKA2"
    cover = fs.find_images_in_dir(os.path.join(data_dir, "Cover"))
    test = fs.find_images_in_dir(os.path.join(data_dir, "Test"))
    all_images = np.array(cover + test)

    hsv_clusters = cluster_images_by_hsv(all_images)

    unique_clusters = np.unique(hsv_clusters)
    for c in unique_clusters:
        make_mosaic(all_images[hsv_clusters == c], plot_images=True)


if __name__ == "__main__":
    main()
