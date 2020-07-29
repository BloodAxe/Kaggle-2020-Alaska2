import argparse
import os

from pytorch_toolbelt.utils import fs
from tqdm import tqdm
from alaska2.dataset import decode_bgr_from_dct
import numpy as np
import cv2


def block8_sum(a: np.ndarray):
    shape = a.shape
    a = a.reshape([a.shape[0] // 8, 8, a.shape[1] // 8, 8] + list(a.shape[2:]))
    a = a.sum(axis=(1, 3))
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))

    args = parser.parse_args()
    data_dir = args.data_dir

    cover = fs.find_images_in_dir(os.path.join(data_dir, "Cover"))
    jimi = fs.find_images_in_dir(os.path.join(data_dir, "JMiPOD"))
    juni = fs.find_images_in_dir(os.path.join(data_dir, "JUNIWARD"))
    uerd = fs.find_images_in_dir(os.path.join(data_dir, "UERD"))

    for cover_fname, jimi_fname, juni_fname, uerd_fname in zip(tqdm(cover), jimi, juni, uerd):
        cover = decode_bgr_from_dct(fs.change_extension(cover_fname, ".npz"))
        jimi = decode_bgr_from_dct(fs.change_extension(jimi_fname, ".npz"))
        juni = decode_bgr_from_dct(fs.change_extension(juni_fname, ".npz"))
        uerd = decode_bgr_from_dct(fs.change_extension(uerd_fname, ".npz"))

        jimi_mask = block8_sum(np.abs(cover - jimi).sum(axis=2)) > 0
        juni_mask = block8_sum(np.abs(cover - juni).sum(axis=2)) > 0
        uerd_mask = block8_sum(np.abs(cover - uerd).sum(axis=2)) > 0

        cover_mask = jimi_mask | juni_mask | uerd_mask

        cv2.imwrite(fs.change_extension(cover_fname, ".png"), cover_mask * 255)
        cv2.imwrite(fs.change_extension(jimi_fname, ".png"), jimi_mask * 255)
        cv2.imwrite(fs.change_extension(juni_fname, ".png"), juni_mask * 255)
        cv2.imwrite(fs.change_extension(uerd_fname, ".png"), uerd_mask * 255)


if __name__ == "__main__":
    main()
