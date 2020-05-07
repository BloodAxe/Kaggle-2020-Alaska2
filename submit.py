import warnings
from collections import defaultdict

from catalyst.utils import any2device
from pytorch_toolbelt.utils.catalyst import report_checkpoint

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import argparse
import os

import cv2
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from alaska2 import *


@torch.no_grad()
def main():
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="+")
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))
    parser.add_argument("-od", "--output-dir", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-w", "--workers", type=int, default=0)
    parser.add_argument("--tta", type=str, default=None)
    parser.add_argument("--activation", type=str, default="after_model")

    args = parser.parse_args()

    checkpoint_fnames = args.checkpoint
    data_dir = args.data_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    workers = args.workers
    tta = args.tta
    activation = args.activation
    features = ["rgb"]

    if output_dir is None:
        dirnames = list(set([os.path.dirname(ck) for ck in checkpoint_fnames]))
        if len(dirnames) == 1:
            output_dir = dirnames[0]
        else:
            raise ValueError(
                "A submission csv file must be specified explicitly since checkpoints exists in various folders"
            )

    submission_csv = os.path.join(output_dir, "submission.csv")

    print("Submission will be saved to ", submission_csv)

    os.makedirs(os.path.dirname(submission_csv), exist_ok=True)

    test_ds = get_test_dataset(data_dir, features=features)
    outputs = [OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_MODIFICATION_TYPE]

    model, checkpoints = ensemble_from_checkpoints(
        checkpoint_fnames, strict=False, outputs=outputs, activation=activation, tta=tta
    )

    for c in checkpoints:
        report_checkpoint(c)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.eval()
    loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=False, drop_last=False
    )

    proposalcsv_flag = defaultdict(list)
    proposalcsv_type = defaultdict(list)
    proposalcsv_type_flag = defaultdict(list)

    for batch in tqdm(loader):
        batch = any2device(batch, device="cuda")
        probas_flag = predict_from_flag(model, batch)
        probas_type = predict_from_type(model, batch)
        probas_flag_type = predict_from_flag_and_type_sum(model, batch)

        for i, image_id in enumerate(batch[INPUT_IMAGE_ID_KEY]):
            proposalcsv_flag["Id"].append(image_id + ".jpg")
            proposalcsv_flag["Label"].append(float(probas_flag[i]))

            proposalcsv_type["Id"].append(image_id + ".jpg")
            proposalcsv_type["Label"].append(float(probas_type[i]))

            proposalcsv_type_flag["Id"].append(image_id + ".jpg")
            proposalcsv_type_flag["Label"].append(float(probas_flag_type[i]))

    proposalcsv = pd.DataFrame.from_dict(proposalcsv_flag)
    proposalcsv.to_csv(os.path.join(output_dir, "submission_flag.csv"), index=False)
    print(proposalcsv.head())

    proposalcsv = pd.DataFrame.from_dict(proposalcsv_type)
    proposalcsv.to_csv(os.path.join(output_dir, "submission_type.csv"), index=False)
    print(proposalcsv.head())

    proposalcsv = pd.DataFrame.from_dict(proposalcsv_type_flag)
    proposalcsv.to_csv(os.path.join(output_dir, "submission_type_flag.csv"), index=False)
    print(proposalcsv.head())


if __name__ == "__main__":
    main()
