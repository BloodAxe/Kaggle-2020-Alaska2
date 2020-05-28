import pickle
import warnings
from collections import defaultdict

from catalyst.utils import any2device
from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.catalyst import report_checkpoint
from sklearn.calibration import calibration_curve

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
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as IR
import matplotlib.pyplot as plt


def sigmoid(x):
    return torch.sigmoid(torch.tensor(x)).item()


def classifier_probas(x):
    x = x.replace("[", "").replace("]", "").split(",")
    x = [float(i) for i in x]
    x = torch.tensor(x).softmax(dim=0)
    x = x[1:].sum().clamp(0, 1)
    return float(x)


@torch.no_grad()
def main():
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("oof_predictions", type=str, nargs="+")
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))
    parser.add_argument("-od", "--output-dir", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-w", "--workers", type=int, default=0)
    parser.add_argument("--tta", type=str, default=None)
    parser.add_argument("--activation", type=str, default="after_model")

    args = parser.parse_args()

    oof_predictions = args.oof_predictions

    for p in oof_predictions:
        df = pd.read_csv(p)

        # df[OUTPUT_PRED_MODIFICATION_FLAG] = df[OUTPUT_PRED_MODIFICATION_FLAG].apply(sigmoid)
        df[OUTPUT_PRED_MODIFICATION_TYPE] = df[OUTPUT_PRED_MODIFICATION_TYPE].apply(classifier_probas)

        print(
            "Uncalibrated", alaska_weighted_auc(df["true_modification_flag"].values, df[OUTPUT_PRED_MODIFICATION_FLAG])
        )
        print(
            "Uncalibrated",
            alaska_weighted_auc(df["true_modification_flag"].values, df[OUTPUT_PRED_MODIFICATION_TYPE].values),
        )

        ir = IR(out_of_bounds="clip")
        ir.fit(df[OUTPUT_PRED_MODIFICATION_FLAG].values, df["true_modification_flag"].values)
        flag_calibrated = ir.transform(df[OUTPUT_PRED_MODIFICATION_FLAG])
        print("IR (flag)", alaska_weighted_auc(df["true_modification_flag"].values, flag_calibrated))

        ir = IR(out_of_bounds="clip")
        ir.fit(df[OUTPUT_PRED_MODIFICATION_TYPE].values, df["true_modification_flag"].values)
        type_calibrated = ir.transform(df[OUTPUT_PRED_MODIFICATION_TYPE])
        print("IR (type)", alaska_weighted_auc(df["true_modification_flag"].values, type_calibrated))
        # with open(os.path.join(output_dir, "calibration.pkl"), "wb") as f:
        #     pickle.dump(ir, f)

        plt.figure(figsize=(10, 10))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        fraction_of_positives, mean_predicted_value = calibration_curve(
            df["true_modification_flag"].values, df[OUTPUT_PRED_MODIFICATION_FLAG].values, normalize=True
        )
        plt.plot(mean_predicted_value, fraction_of_positives, label="Uncalibrated (flag)")

        fraction_of_positives, mean_predicted_value = calibration_curve(
            df["true_modification_flag"].values, df[OUTPUT_PRED_MODIFICATION_TYPE].values
        )
        plt.plot(mean_predicted_value, fraction_of_positives, label="Uncalibrated (type)")

        fraction_of_positives, mean_predicted_value = calibration_curve(
            df["true_modification_flag"].values, flag_calibrated
        )
        plt.plot(mean_predicted_value, fraction_of_positives, label="Calibrated (flag)")

        fraction_of_positives, mean_predicted_value = calibration_curve(
            df["true_modification_flag"].values, type_calibrated
        )
        plt.plot(mean_predicted_value, fraction_of_positives, label="Calibrated (type)")
        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
