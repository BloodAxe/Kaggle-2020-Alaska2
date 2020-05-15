import os
from functools import partial

import pandas as pd
import torch
from plotly.tools import ALTERNATIVE_HISTNORM
from pytorch_toolbelt.utils import logit
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# import matplotlib.pyplot as plt


def temperature_scaling(x, t):
    x = torch.tensor(x)
    x_l = logit(x)
    x_s = torch.sigmoid(x_l * t)
    return float(x_s)


def sigmoid(x):
    return torch.sigmoid(torch.tensor(x)).item()


def classifier_probas(x):
    x = x.replace("[", "").replace("]", "").split(",")
    x = [float(i) for i in x]
    x = torch.tensor(x).softmax(dim=0)
    x = x[1:].sum()
    return float(x)


May08_22_42_rgb_resnet34_fold1_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May08_22_42_rgb_resnet34_fold1/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May07_16_48_rgb_resnet34_fold0_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May07_16_48_rgb_resnet34_fold0/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May09_15_13_rgb_densenet121_fold0_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May09_15_13_rgb_densenet121_fold0_fp16/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May11_08_49_rgb_densenet201_fold3_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May11_08_49_rgb_densenet201_fold3_fp16/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May13_19_06_rgb_skresnext50_32x4d_fold1_fp16/best_test_predictions_flip_hv_tta.csv"
)

May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May12_13_01_rgb_skresnext50_32x4d_fold2_fp16/best_test_predictions_flip_hv_tta.csv"
)

May11_09_46_rgb_skresnext50_32x4d_fold3_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May11_09_46_rgb_skresnext50_32x4d_fold3_fp16/best_test_predictions_flip_hv_tta.csv"
)


def stringify_image_id(x):
    return f"{x:04}.jpg"


output_dir = os.path.dirname(__file__)

predictions = {
    # "May08_22_42_rgb_resnet34_fold1_tta": May08_22_42_rgb_resnet34_fold1_tta,
    # "May07_16_48_rgb_resnet34_fold0_tta": May07_16_48_rgb_resnet34_fold0_tta,
    # "May09_15_13_rgb_densenet121_fold0_fp16_tta": May09_15_13_rgb_densenet121_fold0_fp16_tta,
    # "May11_08_49_rgb_densenet201_fold3_fp16_tta": May11_08_49_rgb_densenet201_fold3_fp16_tta,
    "May13_19_06_rgb_skresnext50_32x4d_fold1": May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta,
    "May12_13_01_rgb_skresnext50_32x4d_fold2": May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta,
    "May11_09_46_rgb_skresnext50_32x4d_fold3": May11_09_46_rgb_skresnext50_32x4d_fold3_fp16_tta,
}

# Approach 1 - Average of binary classifier logits w/o temperature
submission1 = May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission1["Id"] = submission1["Id"].apply(stringify_image_id)
submission1["Label"] = sum([df["pred_modification_flag"].values for df in predictions.values()]) / len(predictions)
submission1["Label"] = submission1["Label"].apply(sigmoid)
print(submission1.head())
submission1.to_csv(os.path.join(output_dir, "all_in_average_logits.csv"), index=None)

# Approach 2 - Average of binary classifier probabilities logits w/o temperature
submission2 = May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission2["Id"] = submission2["Id"].apply(stringify_image_id)
submission2["Label"] = sum([df["pred_modification_flag"].apply(sigmoid).values for df in predictions.values()]) / len(
    predictions
)
print(submission2.head())
submission2.to_csv(os.path.join(output_dir, "all_in_average_probas.csv"), index=None)
