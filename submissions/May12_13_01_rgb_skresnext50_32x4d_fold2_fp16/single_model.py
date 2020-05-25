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
    x = x.replace("[","").replace("]","").split(",")
    x = [float(i) for i in x]
    x = torch.tensor(x).softmax(dim=0)
    x = x[1:].sum()
    return float(x)


May12_13_01_rgb_skresnext50_32x4d_fold2_fp16 = pd.read_csv(
    "/old_models/May12_13_01_rgb_skresnext50_32x4d_fold2_fp16/best_test_predictions.csv"
)
May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta = pd.read_csv(
    "/old_models/May12_13_01_rgb_skresnext50_32x4d_fold2_fp16/best_test_predictions_flip_hv_tta.csv"
)


def stringify_image_id(x):
    return f"{x:04}.jpg"


output_dir = os.path.dirname(__file__)

# Approach 1 - binary classifier predictions
submission1 = May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta.copy()
submission1["Id"] = submission1["image_id"].apply(stringify_image_id)
submission1["Label"] = May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta["pred_modification_flag"].apply(sigmoid)

print(submission1.head())
submission1 = submission1[["Id", "Label"]]
submission1.to_csv(os.path.join(output_dir, "May12_13_01_rgb_skresnext50_32x4d_fold2_tta_binary.csv"), index=None)

# Approach 2 - multiclass classifier predictions
submission2 = May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta.copy()
submission2["Id"] = submission2["image_id"].apply(stringify_image_id)
submission2["Label"] = May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta["pred_modification_type"].apply(
    classifier_probas
)

print(submission2.head())
submission2 = submission2[["Id", "Label"]]
submission2.to_csv(os.path.join(output_dir, "May12_13_01_rgb_skresnext50_32x4d_fold2_tta_classifier.csv"), index=None)
