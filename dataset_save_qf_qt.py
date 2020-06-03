import argparse
import os
from collections import defaultdict

import numpy as np
import cv2
import torch
from pytorch_toolbelt.utils import fs, to_numpy
from tqdm import tqdm

from alaska2 import idct8
from alaska2.dataset import idct8v2
from alaska2.models.dct import SpaceToDepth

import torch
import pandas as pd
from torch import Tensor
from typing import Iterable


def quality_factror_from_qm(qm):
    if qm[0, 0] == 2:
        # print('Quality Factor is 95')
        return 95
    elif qm[0, 0] == 3:
        # print('Quality Factor is 90')
        return 90
    elif qm[0, 0] == 8:
        return 75
    else:
        raise ValueError("Unknown quality factor" + str(qm[0, 0]))


def target_from_fname(image_fname):
    if "Cover" in image_fname:
        return 0
    if "JMiPOD" in image_fname:
        return 1
    if "JUNIWARD" in image_fname:
        return 2
    if "UERD" in image_fname:
        return 3
    raise KeyError(image_fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))

    args = parser.parse_args()
    data_dir = args.data_dir

    cover = os.path.join(data_dir, "Cover")
    JMiPOD = os.path.join(data_dir, "JMiPOD")
    JUNIWARD = os.path.join(data_dir, "JUNIWARD")
    UERD = os.path.join(data_dir, "UERD")

    dataset = (
        fs.find_images_in_dir(cover)
        + fs.find_images_in_dir(JMiPOD)
        + fs.find_images_in_dir(JUNIWARD)
        + fs.find_images_in_dir(UERD)
    )
    # dataset = dataset[:500]
    df = defaultdict(list)
    for image_fname in tqdm(dataset):
        target = target_from_fname(image_fname)
        dct_fname = fs.change_extension(image_fname, ".npz")
        dct_data = np.load(dct_fname)
        qm0 = dct_data["qm0"]
        qm1 = dct_data["qm1"]
        qf = quality_factror_from_qm(qm0)
        fsize = os.stat(image_fname).st_size

        df["image_id"].append(fs.id_from_fname(image_fname))
        df["target"].append(target)
        df["quality"].append(qf)
        df["qm0"].append(qm0.flatten().tolist())
        df["qm1"].append(qm1.flatten().tolist())
        df["file_size"].append(fsize)

    df = pd.DataFrame.from_dict(df)
    df.to_csv("dataset_qf_qt.csv", index=False)


if __name__ == "__main__":
    main()
