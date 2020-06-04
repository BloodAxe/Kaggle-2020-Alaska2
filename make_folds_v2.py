import argparse
import os
from collections import defaultdict

import pandas as pd
import numpy as np
from pytorch_toolbelt.utils import fs
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from alaska2 import INPUT_IMAGE_ID_KEY, INPUT_FOLD_KEY


def stringify_image_id(x):
    return f"{x:05}.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))

    args = parser.parse_args()

    df = pd.read_csv("dataset_qf_qt.csv")

    cover_only = df[df["target"] == 0].copy()
    cover_only[INPUT_FOLD_KEY] = -1

    cover_only["quality"] = LabelEncoder().fit_transform(cover_only.quality)
    file_size = cover_only.file_size.values
    file_size_bins = np.digitize(file_size, np.linspace(0, file_size.max(), 4), right=True)
    a = np.bincount(file_size_bins)

    stratify = file_size_bins * 4 + cover_only["quality"].values

    train_val_idx, holdout_idx = train_test_split(
        np.arange(len(cover_only)), stratify=stratify, test_size=5000, shuffle=True, random_state=42
    )

    num_folds = 4

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1234)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_idx, stratify[train_val_idx])):
        cover_only.loc[train_val_idx[val_idx], INPUT_FOLD_KEY] = fold

    cover_only[INPUT_IMAGE_ID_KEY] = cover_only[INPUT_IMAGE_ID_KEY].apply(stringify_image_id)
    cover_only.to_csv("folds_v2.csv", index=False)

    folds = np.unique(cover_only[INPUT_FOLD_KEY])
    for f in folds:
        fold_view = cover_only[cover_only.fold == f]
        print(f, len(fold_view), np.bincount(fold_view.quality))


if __name__ == "__main__":
    main()
