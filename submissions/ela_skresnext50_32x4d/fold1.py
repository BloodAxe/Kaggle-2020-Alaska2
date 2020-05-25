import os

import pandas as pd
import torch
from pytorch_toolbelt.utils import logit


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


def stringify_image_id(x):
    return f"{x:0>4}.jpg"


# May15_17_03_ela_skresnext50_32x4d_fold1 = pd.read_csv(
#     "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_test_predictions.csv"
# )
#
# May15_17_03_ela_skresnext50_32x4d_fold1_flip_tta = pd.read_csv(
#     "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_test_predictions_flip_hv_tta.csv"
# )

May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_test_predictions_d4_tta.csv"
)

May15_17_03_ela_skresnext50_32x4d_fold1_best_loss = pd.read_csv(
    "D:/Develop/Kaggle/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best_loss_test_predictions_flip_hv_tta.csv"
)


output_dir = os.path.dirname(__file__)


# # Approach 1 - No TTA, binary classifier
# submission1 = May15_17_03_ela_skresnext50_32x4d_fold1.copy().rename(columns={"image_id": "Id"})[["Id"]]
# submission1["Id"] = submission1["Id"].apply(stringify_image_id)
# submission1["Label"] = May15_17_03_ela_skresnext50_32x4d_fold1["pred_modification_flag"].apply(sigmoid)
# print(submission1.head())
# submission1.to_csv(os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1.csv"), index=None)
#
# # Approach 2 - Flip TTA, binary classifier
# submission2 = May15_17_03_ela_skresnext50_32x4d_fold1_flip_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
# submission2["Id"] = submission2["Id"].apply(stringify_image_id)
# submission2["Label"] = May15_17_03_ela_skresnext50_32x4d_fold1_flip_tta["pred_modification_flag"].apply(sigmoid)
# print(submission2.head())
# submission2.to_csv(os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_flips_tta.csv"), index=None)

# Approach 3 - D4 TTA, binary classifier
submission1 = May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission1["Id"] = submission1["Id"].apply(stringify_image_id)
submission1["Label"] = May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta["pred_modification_flag"].apply(sigmoid)
print(submission1.head())
submission1.to_csv(os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta_binary.csv"), index=None)

submission2 = May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission2["Id"] = submission2["Id"].apply(stringify_image_id)
submission2["Label"] = May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta["pred_modification_type"].apply(
    classifier_probas
)
print(submission2.head())
submission2.to_csv(
    os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta_classifier.csv"), index=None
)

submission3 = May15_17_03_ela_skresnext50_32x4d_fold1_best_loss.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission3["Id"] = submission3["Id"].apply(stringify_image_id)
submission3["Label"] = May15_17_03_ela_skresnext50_32x4d_fold1_best_loss["pred_modification_flag"].apply(sigmoid)
print(submission3.head())
submission3.to_csv(
    os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_best_loss_flip_tta_binary.csv"), index=None
)

predictions = {
    "May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta": May15_17_03_ela_skresnext50_32x4d_fold1_d4_tta,
    "May15_17_03_ela_skresnext50_32x4d_fold1_best_loss": May15_17_03_ela_skresnext50_32x4d_fold1_best_loss,
}

# Submission 4
submission4 = May15_17_03_ela_skresnext50_32x4d_fold1_best_loss.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission4["Id"] = submission4["Id"].apply(stringify_image_id)
submission4["Label"] = sum([df["pred_modification_flag"].apply(sigmoid).values for df in predictions.values()]) / len(
    predictions
)
print(submission4.head())
submission4.to_csv(
    os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_avg_best_auc_best_loss_binary.csv"), index=None
)

# Submission 5
submission5 = May15_17_03_ela_skresnext50_32x4d_fold1_best_loss.copy().rename(columns={"image_id": "Id"})[["Id"]]
submission5["Id"] = submission5["Id"].apply(stringify_image_id)
submission5["Label"] = sum(
    [df["pred_modification_type"].apply(classifier_probas).values for df in predictions.values()]
) / len(predictions)
print(submission5.head())
submission5.to_csv(
    os.path.join(output_dir, "May15_17_03_ela_skresnext50_32x4d_fold1_avg_best_auc_best_loss_classifier.csv"),
    index=None,
)
