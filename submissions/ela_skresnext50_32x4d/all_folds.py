import os
import pandas as pd
import torch
from pytorch_toolbelt.utils import logit
from typing import List, Union

from alaska2 import OUTPUT_PRED_MODIFICATION_FLAG, alaska_weighted_auc, OUTPUT_PRED_MODIFICATION_TYPE

output_dir = os.path.dirname(__file__)


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


def submit_from_binary_calibrated(test_predictions: List[str], oof_predictions: List[str]):
    preds_df = [calibrated(pd.read_csv(x), pd.read_csv(y)) for x, y in zip(test_predictions, oof_predictions)]

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_flag"].values for df in preds_df]) / len(preds_df)
    return submission


def submit_from_classifier_calibrated(test_predictions: List[str], oof_predictions: List[str]):
    preds_df = [calibrated(pd.read_csv(x), pd.read_csv(y)) for x, y in zip(test_predictions, oof_predictions)]

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_type"].values for df in preds_df]) / len(preds_df)
    return submission


def calibrated(test_predictions, oof_predictions):
    """
    Update test predictions w.r.t to calibration trained on OOF predictions
    :param test_predictions:
    :param oof_predictions:
    :return:
    """
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.isotonic import IsotonicRegression as IR

    oof_predictions = oof_predictions.copy()
    oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].apply(
        classifier_probas
    )

    test_predictions = test_predictions.copy()
    test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].apply(
        classifier_probas
    )

    y_true = oof_predictions["true_modification_flag"].values
    ir_flag = IR(out_of_bounds="clip")
    ir_flag.fit(oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values, y_true)

    ir_type = IR(out_of_bounds="clip")
    ir_type.fit(oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values, y_true)

    flag_calibrated = ir_flag.transform(oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values)
    type_calibrated = ir_flag.transform(oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values)
    print("Calibration results")
    print(
        "Flag",
        alaska_weighted_auc(y_true, oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values),
        alaska_weighted_auc(y_true, flag_calibrated),
    )
    print(
        "Type",
        alaska_weighted_auc(y_true, oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values),
        alaska_weighted_auc(y_true, type_calibrated),
    )

    test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = ir_flag.transform(
        test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
    )
    test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = ir_type.transform(
        test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
    )
    return test_predictions


best_loss = [
    "models/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints/best_test_predictions.csv",
    "models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints/best_test_predictions.csv",
    "models/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints/best_test_predictions.csv",
    "models/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints/best_test_predictions.csv",
]

best_loss_oof = [
    "models/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints/best_oof_predictions.csv",
    "models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints/best_oof_predictions.csv",
    "models/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints/best_oof_predictions.csv",
    "models/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints/best_oof_predictions.csv",
]


best_auc_b = [
    "models/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc/best_test_predictions.csv",
    "models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc/best_test_predictions.csv",
    "models/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc/best_test_predictions.csv",
    "models/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc/best_test_predictions.csv",
]

best_auc_b_oof = [
    "models/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc/best_oof_predictions.csv",
    "models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc/best_oof_predictions.csv",
    "models/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc/best_oof_predictions.csv",
    "models/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc/best_oof_predictions.csv",
]

best_auc_c = [
    "models/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
    "models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
    "models/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
    "models/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
]

best_auc_c_oof = [
    "models/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
    "models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
    "models/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
    "models/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
]


submit_from_average_binary(best_loss).to_csv(
    os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_loss_binary.csv"), index=None
)
submit_from_average_classifier(best_loss).to_csv(
    os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_loss_classifier.csv"), index=None
)
submit_from_average_binary(best_auc_b).to_csv(
    os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_auc_b.csv"), index=None
)
submit_from_average_classifier(best_auc_c).to_csv(
    os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_auc_c.csv"), index=None
)

submit_from_classifier_calibrated(best_auc_c, best_auc_c_oof).to_csv(
    os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_auc_c_calibrated.csv"), index=None
)
