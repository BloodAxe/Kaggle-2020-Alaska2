import os

import pandas as pd
import numpy as np


def infer_fold(x):
    if "fold0" in x:
        return 0
    if "fold1" in x:
        return 1
    if "fold2" in x:
        return 2
    if "fold3" in x:
        return 3

    raise KeyError(x)


import os, sys


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def main():
    df = pd.read_csv("summary.csv")
    df["fold"] = df["test_predictions"].apply(infer_fold)

    keys = ["b_auc_before", "b_auc_after", "c_auc_before", "c_auc_after"]

    index = np.argmax(df[keys].values, 1)
    val = np.max(df[keys].values, 1)
    df["max_column"] = np.array(keys)[index]
    df["max_auc"] = val

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    df["test_predictions"] = df["test_predictions"].apply(lambda x: splitall(x)[1])

    for fold in range(4):
        fold_df = df[df.fold == fold].sort_values(by="max_auc", ascending=False)

        fold_df = fold_df[["test_predictions", "checkpoint_metric", "tta", "max_column", "max_auc"]]
        print("Fold", fold)
        print(fold_df.head(3))


if __name__ == "__main__":
    main()
