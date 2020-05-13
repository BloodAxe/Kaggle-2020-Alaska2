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


May08_22_42_rgb_resnet34_fold1 = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May08_22_42_rgb_resnet34_fold1/main/checkpoints_auc/best_test_predictions.csv"
)
May08_22_42_rgb_resnet34_fold1_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May08_22_42_rgb_resnet34_fold1/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May07_16_48_rgb_resnet34_fold0 = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May07_16_48_rgb_resnet34_fold0/main/checkpoints_auc/best_test_predictions.csv"
)
May07_16_48_rgb_resnet34_fold0_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May07_16_48_rgb_resnet34_fold0/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May09_15_13_rgb_densenet121_fold0_fp16 = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May09_15_13_rgb_densenet121_fold0_fp16/main/checkpoints_auc/best_test_predictions.csv"
)
May09_15_13_rgb_densenet121_fold0_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May09_15_13_rgb_densenet121_fold0_fp16/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May11_08_49_rgb_densenet201_fold3_fp16 = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May11_08_49_rgb_densenet201_fold3_fp16/main/checkpoints_auc/best_test_predictions.csv"
)
May11_08_49_rgb_densenet201_fold3_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May11_08_49_rgb_densenet201_fold3_fp16/main/checkpoints_auc/best_test_predictions_flip_hv_tta.csv"
)

May11_09_46_rgb_skresnext50_32x4d_fold3_fp16 = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May11_09_46_rgb_skresnext50_32x4d_fold3_fp16/best_test_predictions.csv"
)
May11_09_46_rgb_skresnext50_32x4d_fold3_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May11_09_46_rgb_skresnext50_32x4d_fold3_fp16/best_test_predictions_flip_hv_tta.csv"
)

May12_13_01_rgb_skresnext50_32x4d_fold2_fp16 = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May12_13_01_rgb_skresnext50_32x4d_fold2_fp16/best_test_predictions.csv"
)
May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May12_13_01_rgb_skresnext50_32x4d_fold2_fp16/best_test_predictions_flip_hv_tta.csv"
)

predictions = {
    # "May07_16_48_rgb_resnet34_fold0": May07_16_48_rgb_resnet34_fold0_tta,
    # "May08_22_42_rgb_resnet34_fold1": May08_22_42_rgb_resnet34_fold1_tta,
    "May09_15_13_rgb_densenet121_fold0": May09_15_13_rgb_densenet121_fold0_fp16_tta,
    "May11_08_49_rgb_densenet201_fold3": May11_08_49_rgb_densenet201_fold3_fp16_tta,
    "May11_09_46_rgb_skresnext50_32x4d_fold3": May11_09_46_rgb_skresnext50_32x4d_fold3_fp16_tta,
    "May12_13_01_rgb_skresnext50_32x4d_fold2": May12_13_01_rgb_skresnext50_32x4d_fold2_fp16_tta,
}


# Group data together
# hist_data = []
# group_labels = []
#
# for name, p in predictions.items():
#     hist_data.append(p["pred_modification_flag"].values)
#     group_labels.append(name)
#
# # Create distplot with custom bin_size
# fig = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
# fig.update_layout(title_text="Hist and Curve Plot")
# fig.show()

def stringify_image_id(x):
    return f"{x:04}.jpg"

output_dir = os.path.dirname(__file__)

# Approach 1 - Average of binary classifier logits w/o temperature
submission1 = May07_16_48_rgb_resnet34_fold0.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission1["Id"] = submission1["Id"].apply(stringify_image_id)
submission1["Label"] = sum([df["pred_modification_flag"].values for df in predictions.values()]) / len(predictions)
submission1["Label"] = submission1["Label"].apply(sigmoid)
print(submission1.head())
submission1.to_csv(os.path.join(output_dir, "six_models_average_logits.csv"), index=None)

# Approach 2 - Average of binary classifier probabilities logits w/o temperature
submission2 = May07_16_48_rgb_resnet34_fold0.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission2["Id"] = submission2["Id"].apply(stringify_image_id)
submission2["Label"] = sum([df["pred_modification_flag"].apply(sigmoid).values for df in predictions.values()]) / len(
    predictions
)
print(submission2.head())
submission2.to_csv(os.path.join(output_dir, "six_models_average_probas.csv"), index=None)


# Approach 3 - Average of binary classifier logits with temperature
temperature = 1.0 / 36.0
submission3 = May07_16_48_rgb_resnet34_fold0.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission3["Id"] = submission3["Id"].apply(stringify_image_id)
submission3["Label"] = sum([df["pred_modification_flag"].values for df in predictions.values()]) * temperature
submission3["Label"] = submission3["Label"].apply(sigmoid)
print(submission3.head())
submission3.to_csv(os.path.join(output_dir, "six_models_average_logits_temp_1_36.csv"), index=None)

# Approach 4 - Average of binary classifier logits with temperature
temperature = 1.0 / 2.0
submission4 = May07_16_48_rgb_resnet34_fold0.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission4["Id"] = submission4["Id"].apply(stringify_image_id)
submission4["Label"] = sum([df["pred_modification_flag"].values for df in predictions.values()]) * temperature
submission4["Label"] = submission4["Label"].apply(sigmoid)
print(submission4.head())
submission4.to_csv(os.path.join(output_dir, "six_models_average_logits_temp_1_2.csv"), index=None)

# Create distplot with custom bin_size
fig = ff.create_distplot(
    [submission1["Label"], submission2["Label"], submission3["Label"], submission4["Label"]],
    ["submission1", "submission2", "submission3", "submission4"],
    bin_size=0.2,
)
fig.update_layout(title_text="Hist and Curve Plot")
fig.show()
