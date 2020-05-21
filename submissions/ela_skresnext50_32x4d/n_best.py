import os

import pandas as pd
import torch
from pytorch_toolbelt.utils import logit


output_dir = os.path.dirname(__file__)

# import matplotlib.pyplot as plt
from typing import List


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


def stringify_image_id(x):
    return f"{x:0>4}.jpg"


fold1_best_5_predictions = [
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.66_test_predictions_flip_hv_tta.csv",
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.70_test_predictions_flip_hv_tta.csv",
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.71_test_predictions_flip_hv_tta.csv",
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.73_test_predictions_flip_hv_tta.csv",
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.74_test_predictions_flip_hv_tta.csv",
]


def submit_from_average_binary(preds: List[str]):
    preds_df = [pd.read_csv(x) for x in preds]

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_flag"].apply(sigmoid).values for df in preds_df]) / len(preds_df)
    return submission


def submit_from_average_classifier(preds: List[str]):
    preds_df = [pd.read_csv(x) for x in preds]

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_type"].apply(classifier_probas).values for df in preds_df]) / len(
        preds_df
    )
    return submission


sub1 = submit_from_average_classifier(fold1_best_5_predictions)
print(sub1.head())
sub1.to_csv(os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_best_5_classifier.csv"), index=None)


sub2 = submit_from_average_binary(fold1_best_5_predictions)
print(sub2.head())
sub2.to_csv(os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_best_5_binary.csv"), index=None)

# All shit together
sub3 = submit_from_average_classifier(
    [
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.66_test_predictions_d4_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.70_test_predictions_d4_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.71_test_predictions_d4_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.73_test_predictions_d4_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/auc/train.74_test_predictions_d4_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_loss_test_predictions_d4_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_test_predictions_d4_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_test_predictions_flip_hv_tta.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_test_predictions.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_loss_test_predictions.csv",
        "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_loss_test_predictions_flip_hv_tta.csv",
    ]
)
print(sub3.head())
sub3.to_csv(os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_all_shit_together.csv"), index=None)
