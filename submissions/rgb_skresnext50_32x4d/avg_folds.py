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


May13_23_00_rgb_skresnext50_32x4d_fold0_fp16_tta = pd.read_csv(
    "/old_models/May13_23_00_rgb_skresnext50_32x4d_fold0_fp16/best_test_predictions_flip_hv_tta.csv"
)

May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta = pd.read_csv(
    "/old_models/May13_19_06_rgb_skresnext50_32x4d_fold1_fp16/best_test_predictions_flip_hv_tta.csv"
)

May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta = pd.read_csv(
    "/old_models/May12_13_01_rgb_skresnext50_32x4d_fold2_fp16/best_test_predictions_flip_hv_tta.csv"
)

May11_09_46_rgb_skresnext50_32x4d_fold3_fp16_tta = pd.read_csv(
    "/old_models/May11_09_46_rgb_skresnext50_32x4d_fold3_fp16/best_test_predictions_flip_hv_tta.csv"
)

# ELA
May15_17_03_ela_skresnext50_32x4d_fold1_flips_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_test_predictions_flip_hv_tta.csv"
)


def stringify_image_id(x):
    return f"{x:04}.jpg"


output_dir = os.path.dirname(__file__)

predictions = {
    "May13_23_00_rgb_skresnext50_32x4d_fold0": May13_23_00_rgb_skresnext50_32x4d_fold0_fp16_tta,
    "May13_19_06_rgb_skresnext50_32x4d_fold1": May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta,
    "May12_13_01_rgb_skresnext50_32x4d_fold2": May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta,
    # "May11_09_46_rgb_skresnext50_32x4d_fold3": May11_09_46_rgb_skresnext50_32x4d_fold3_fp16_tta,
    "May15_17_03_ela_skresnext50_32x4d_fold1_flips": May15_17_03_ela_skresnext50_32x4d_fold1_flips_tta,
}

# # Approach 1 - Average of binary classifier logits w/o temperature
# submission1 = May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
# submission1["Id"] = submission1["Id"].apply(stringify_image_id)
# submission1["Label"] = sum([df["pred_modification_flag"].values for df in predictions.values()]) / len(predictions)
# submission1["Label"] = submission1["Label"].apply(sigmoid)
# print(submission1.head())
# submission1.to_csv(os.path.join(output_dir, "rgb_skresnext50_32x4d_logits.csv"), index=None)


# Approach 2 - Geometric average of binary classifier probabilities
submission2 = May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission2["Id"] = submission2["Id"].apply(stringify_image_id)
submission2["Label"] = np.power(
    np.prod([df["pred_modification_flag"].apply(sigmoid).values for df in predictions.values()],axis=0),
    1.0 / len(predictions),
)
print(submission2.head())
submission2.to_csv(os.path.join(output_dir, "rgb_skresnext50_32x4d_probas_wo_fold3_geom.csv"), index=None)

# Approach 3 - Harmonic mean of binary classifier probabilities
submission3 = May13_19_06_rgb_skresnext50_32x4d_fold1_fp16_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission3["Id"] = submission3["Id"].apply(stringify_image_id)
submission3["Label"] = 1.0 / sum(
    [1.0 / df["pred_modification_flag"].apply(sigmoid).values for df in predictions.values()]
)
print(submission3.head())
submission3.to_csv(os.path.join(output_dir, "rgb_skresnext50_32x4d_probas_wo_fold3_hmean.csv"), index=None)
