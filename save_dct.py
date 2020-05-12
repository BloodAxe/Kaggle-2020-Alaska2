import argparse
from functools import partial
from multiprocessing import Pool

from pytorch_toolbelt.utils import fs, os
from tqdm import tqdm

from alaska2 import compute_dct
import numpy as np

def extract_and_save_dct(fname, output_dir):
    dct_y, dct_cr, dct_cb = compute_dct(fname)

    image_id = fs.id_from_fname(fname) + ".npz"
    method = os.path.split(fname)[-2]
    dct_fname = os.path.join(output_dir, method, image_id)
    np.savez(dct_fname, dct_y=dct_y, dct_cr=dct_cr, dct_cb=dct_cb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))
    parser.add_argument("-od", "--output-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))

    args = parser.parse_args()

    data_dir = args.data_dir
    original_images = (
        fs.find_images_in_dir(os.path.join(data_dir, "Cover"))
        + fs.find_images_in_dir(os.path.join(data_dir, "JMiPOD"))
        + fs.find_images_in_dir(os.path.join(data_dir, "JUNIWARD"))
        + fs.find_images_in_dir(os.path.join(data_dir, "UERD"))
    )
    os.makedirs(args.output_dir, exist_ok=True)

    process_fn = partial(extract_and_save_dct, output_dir=args.output_dir)
    with Pool(8) as wp:
        for _ in tqdm(wp.imap_unordered(process_fn, original_images)):
            pass


if __name__ == "__main__":
    main()