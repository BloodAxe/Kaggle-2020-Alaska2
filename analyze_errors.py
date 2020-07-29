import argparse
import math
import os
from multiprocessing import Pool

import cv2
import pandas as pd
from collections import defaultdict

from pytorch_toolbelt.utils import fs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def main():
    methods = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    errors = pd.read_csv("errors.csv")
    errors["method"] = errors["true_modification_type"].apply(lambda x: methods[int(x)])
    errors["key"] = errors["image_id"] + "_" + errors["method"]
    errors = errors[["key", "error", "true_modification_type", "method"]]

    changed_bits = pd.read_csv("changed_bits.csv")
    changed_bits["key"] = changed_bits["file"] + "_" + changed_bits["method"]
    changed_bits = changed_bits[changed_bits["method"] != "Cover"]
    changed_bits = changed_bits[["key", "nbits"]]
    # df = pd.merge(errors, changed_bits, on="key")

    analyze_embeddings = pd.read_csv("analyze_embeddings.csv")
    analyze_embeddings["key"] = analyze_embeddings["image"] + "_" + analyze_embeddings["method"]
    analyze_embeddings["nbits"] = analyze_embeddings["dct_total"]
    # analyze_embeddings = analyze_embeddings[analyze_embeddings["method"] != "Cover"]

    analyze_embeddings = analyze_embeddings[["key", "nbits"]]

    df = pd.merge(errors, analyze_embeddings, on="key")
    print(len(df), df.head())

    df = df[df["error"] > 1e-3]

    fig, ax = plt.subplots(figsize=(20, 8))
    for idx, method in enumerate(["JMiPOD", "JUNIWARD", "UERD"]):
        m = df[df["method"] == method]
        ax.scatter(np.log(m["nbits"].values), m["error"].values, label=method, alpha=0.3, edgecolors="none")

    ax.legend()
    ax.set_xlabel("log(nbits)")
    ax.set_ylabel("error")
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(20, 8))
    m = errors[errors["method"] == "Cover"]
    ax.scatter(np.random.randint(0, 100, len(m)), m["error"].values, label="Cover", alpha=0.3, edgecolors="none")
    ax.legend()
    ax.set_ylabel("error")
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
