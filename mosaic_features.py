import os
from multiprocessing import Pool

import cv2
from pytorch_toolbelt.utils import fs
from tqdm import tqdm
import pandas as pd


def extract_featues(image_fname):
    image = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE)
    image = cv2.pyrDown(image)

    t = image[0, :].flatten().tolist()
    b = image[-1, :].flatten().tolist()

    l = image[:, 0].flatten().tolist()
    r = image[:, -1].flatten().tolist()

    return {"image_fname": image_fname, "top": t, "left": l, "right": r, "bottom": b}




def main():
    data_dir = "D:\datasets\ALASKA2"
    cover = fs.find_images_in_dir(os.path.join(data_dir, "Cover"))
    test = fs.find_images_in_dir(os.path.join(data_dir, "Test"))

    all_images = cover + test

    # all_images = all_images[:100]

    all_features = []
    with Pool(8) as wp:
        for f in tqdm(wp.imap_unordered(extract_featues, all_images), total=len(all_images)):
            all_features.append(f)

    df = pd.DataFrame.from_records(all_features)
    df.to_csv("features.csv", index=False)


if __name__ == "__main__":
    main()
