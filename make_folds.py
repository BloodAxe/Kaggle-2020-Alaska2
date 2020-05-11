import argparse
import os
from collections import defaultdict

import pandas as pd
import numpy as np
from pytorch_toolbelt.utils import fs

from alaska2 import INPUT_IMAGE_ID_KEY, INPUT_FOLD_KEY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))

    args = parser.parse_args()

    data_dir = args.data_dir

    original_images = np.array(fs.find_images_in_dir(os.path.join(data_dir, "Cover")))
    image_sizes = np.array([os.stat(fname).st_size for fname in original_images])
    order = np.argsort(image_sizes)
    original_images = original_images[order]
    num_folds = 4
    num_images = len(original_images)

    folds_lut = (list(range(num_folds)) * num_images)[:num_images]
    folds_lut = np.array(folds_lut)

    df = defaultdict(list)
    df[INPUT_IMAGE_ID_KEY].extend([os.path.basename(x) for x in original_images])
    df[INPUT_FOLD_KEY].extend(folds_lut)

    df = pd.DataFrame.from_dict(df)
    df.to_csv("folds.csv", index=False)


if __name__ == "__main__":
    main()
