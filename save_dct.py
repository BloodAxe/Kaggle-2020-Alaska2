import argparse
import gc
from functools import partial
from multiprocessing import Pool

from pytorch_toolbelt.utils import fs, os
from tqdm import tqdm

from alaska2 import compute_dct_slow, compute_dct_fast
import numpy as np
import jpegio as jpio


# ! git clone https://github.com/dwgoon/jpegio
# # Once downloaded install the package
# !pip install jpegio/.


def extract_and_save_dct_uber(fname, output_dir):
    dct_y, dct_cb, dct_cr = compute_dct_fast(fname)

    image_id = fs.id_from_fname(fname) + ".npz"
    method = os.path.split(os.path.split(fname)[0])[1]
    dct_fname = os.path.join(output_dir, method, image_id)

    np.savez_compressed(dct_fname,
                        dct_y=dct_y,
                        dct_cb=dct_cb,
                        dct_cr=dct_cr)


def extract_and_save_dct_jpegio(fname, output_dir):
    # dct_y, dct_cr, dct_cb = compute_dct_fast(fname)

    image_id = fs.id_from_fname(fname) + ".npz"
    method = os.path.split(os.path.split(fname)[0])[1]
    dct_fname = os.path.join(output_dir, method, image_id)

    jpegStruct = jpio.read(fname)
    dct_matrix = jpegStruct.coef_arrays
    quant_tables = jpegStruct.quant_tables
    # ci0 = jpegStruct.comp_info[0]
    # ci1 = jpegStruct.comp_info[1]
    # ci2 = jpegStruct.comp_info[2]

    qm0 = np.tile(quant_tables[0], (512 // 8, 512 // 8))
    qm1 = np.tile(quant_tables[1], (512 // 8, 512 // 8))
    np.savez_compressed(dct_fname,
                        dct_y=(dct_matrix[0] * qm0).astype(np.int16),
                        dct_cb=(dct_matrix[1] * qm1).astype(np.int16),
                        dct_cr=(dct_matrix[2] * qm1).astype(np.int16),
                        qm0=quant_tables[0].astype(np.int16),
                        qm1=quant_tables[1].astype(np.int16))

    del jpegStruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))
    parser.add_argument("-od", "--output-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))
    parser.add_argument("-f", "--folder", type=str, default=None)
    parser.add_argument("-p", "--part", type=int, default=None)

    args = parser.parse_args()

    data_dir = args.data_dir
    if args.folder is None:
        original_images = (
                fs.find_images_in_dir(os.path.join(data_dir, "Cover"))
                + fs.find_images_in_dir(os.path.join(data_dir, "JMiPOD"))
                + fs.find_images_in_dir(os.path.join(data_dir, "JUNIWARD"))
                + fs.find_images_in_dir(os.path.join(data_dir, "UERD"))
                + fs.find_images_in_dir(os.path.join(data_dir, "Test"))
        )
    else:
        original_images = fs.find_images_in_dir(os.path.join(data_dir, args.folder))
        if args.part is not None:
            half = len(original_images) // 2
            if args.part == 0:
                original_images = original_images[:half]
                print("First half")
            else:
                original_images = original_images[half:]
                print("Second half")

        print(original_images[0])

    os.makedirs(args.output_dir, exist_ok=True)
    process_fn = partial(extract_and_save_dct_jpegio, output_dir=args.output_dir)
    with Pool(16) as wp:
        for _ in tqdm(wp.imap_unordered(process_fn, original_images), total=len(original_images)):
            pass


if __name__ == "__main__":
    main()
