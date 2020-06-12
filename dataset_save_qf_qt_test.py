import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import fs
from tqdm import tqdm


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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))

    args = parser.parse_args()
    data_dir = args.data_dir

    test_dir = os.path.join(data_dir, "Test")
    dataset = fs.find_images_in_dir(test_dir)

    # dataset = dataset[:500]
    df = defaultdict(list)
    for image_fname in tqdm(dataset):
        dct_fname = fs.change_extension(image_fname, ".npz")
        dct_data = np.load(dct_fname)
        qm0 = dct_data["qm0"]
        qm1 = dct_data["qm1"]
        qf = quality_factror_from_qm(qm0)
        fsize = os.stat(image_fname).st_size

        df["image_id"].append(fs.id_from_fname(image_fname))
        df["quality"].append(qf)
        df["qm0"].append(qm0.flatten().tolist())
        df["qm1"].append(qm1.flatten().tolist())
        df["file_size"].append(fsize)

    df = pd.DataFrame.from_dict(df)
    df.to_csv("test_dataset_qf_qt.csv", index=False)


if __name__ == "__main__":
    main()
