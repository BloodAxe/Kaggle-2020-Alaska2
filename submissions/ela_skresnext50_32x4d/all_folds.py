import os
import pandas as pd
import torch
from pytorch_toolbelt.utils import logit
from typing import List, Union

from sklearn.calibration import CalibratedClassifierCV
from torch.nn import functional as F
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


def classifier_probas_raw(x):
    x = x.replace("[", "").replace("]", "").split(",")
    x = [float(i) for i in x]
    x = torch.tensor(x)
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


def submit_from_median_classifier(test_predictions: List[str]):
    preds_df = [pd.read_csv(x) for x in test_predictions]

    p = np.stack([df["pred_modification_type"].apply(classifier_probas).values for df in preds_df])
    p = np.median(p, axis=0)

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = p
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


def submit_from_sum_calibrated(test_predictions: List[str], oof_predictions: List[str]):
    preds_df = [calibrated_sum(pd.read_csv(x), pd.read_csv(y)) for x, y in zip(test_predictions, oof_predictions)]

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_type"].values for df in preds_df]) / len(preds_df)
    return submission


def noop(x):
    return x


def calibrated(test_predictions, oof_predictions, flag_transform=noop, type_transform=classifier_probas):
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
        type_transform
    )
    oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].apply(
        flag_transform
    )

    test_predictions = test_predictions.copy()
    test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].apply(
        type_transform
    )
    test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].apply(
        flag_transform
    )

    y_true = oof_predictions["true_modification_flag"].values

    print("Calibration results")

    if True:
        ir_flag = IR(out_of_bounds="clip")
        ir_flag.fit(oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values, y_true)
        flag_calibrated = ir_flag.transform(oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values)
        score_flag_before = alaska_weighted_auc(y_true, oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values)
        score_flag_after = alaska_weighted_auc(y_true, flag_calibrated)
        print(
            "Flag", score_flag_before, score_flag_after, (score_flag_after - score_flag_before),
        )
        test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = ir_flag.transform(
            test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
        )

    if True:
        ir_type = IR(out_of_bounds="clip")
        ir_type.fit(oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values, y_true)
        type_calibrated = ir_type.transform(oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values)
        score_type_before = alaska_weighted_auc(y_true, oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values)
        score_type_after = alaska_weighted_auc(y_true, type_calibrated)
        print(
            "Type", score_type_before, score_type_after, score_type_after - score_type_before,
        )
        test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = ir_type.transform(
            test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
        )

    return test_predictions


import numpy as np


def calibrated_sum(test_predictions, oof_predictions, flag_transform=sigmoid, type_transform=classifier_probas):
    """
    Update test predictions w.r.t to calibration trained on OOF predictions
    :param test_predictions:
    :param oof_predictions:
    :return:
    """
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.isotonic import IsotonicRegression as IR

    calib_clf = CalibratedClassifierCV(method="isotonic")

    oof_predictions = oof_predictions.copy()
    oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].apply(
        type_transform
    )
    oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].apply(
        flag_transform
    )

    test_predictions = test_predictions.copy()
    test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].apply(
        type_transform
    )
    test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].apply(
        flag_transform
    )

    y_true = oof_predictions["true_modification_flag"].values.astype(int)

    print("Calibration results")

    if True:
        x_uncalibrated = (
            oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
            + oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
        )

        calib_clf.fit(x_uncalibrated.reshape(-1, 1), y_true)
        x_calibrated = calib_clf.predict_proba(x_uncalibrated.reshape(-1, 1))[:, 1]
        score_flag_before = alaska_weighted_auc(y_true, x_uncalibrated)
        score_flag_after = alaska_weighted_auc(y_true, x_calibrated)
        print(
            "Sum", score_flag_before, score_flag_after, (score_flag_after - score_flag_before),
        )
        x_test = (
            test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
            + test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
        )
        x_pred = calib_clf.predict_proba(x_test.reshape(-1, 1))[:, 1]
        test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = x_pred

    return test_predictions


def calibrated_raw(test_predictions, oof_predictions, flag_transform=noop, type_transform=classifier_probas):
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
        type_transform
    )
    oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].apply(
        flag_transform
    )

    test_predictions = test_predictions.copy()
    test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].apply(
        type_transform
    )
    test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].apply(
        flag_transform
    )

    y_true = oof_predictions["true_modification_flag"].values.astype(int)

    print("Calibration results")

    if True:
        x_uncalibrated = (
            oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
            + oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
        )

        calib_clf = IR(out_of_bounds="clip")
        calib_clf.fit(x_uncalibrated, y_true)
        x_calibrated = calib_clf.transform(x_uncalibrated)
        score_flag_before = alaska_weighted_auc(y_true, x_uncalibrated)
        score_flag_after = alaska_weighted_auc(y_true, x_calibrated)
        print(
            "Sum", score_flag_before, score_flag_after, (score_flag_after - score_flag_before),
        )
        x_test = (
            test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
            + test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
        )
        x_pred = calib_clf.transform(x_test)
        test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = x_pred

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

#
# submit_from_average_binary(best_loss).to_csv(
#     os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_loss_binary.csv"), index=None
# )
# submit_from_average_classifier(best_loss).to_csv(
#     os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_loss_classifier.csv"), index=None
# )
# submit_from_average_binary(best_auc_b).to_csv(
#     os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_auc_b.csv"), index=None
# )
# submit_from_average_classifier(best_auc_c).to_csv(
#     os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_auc_c.csv"), index=None
# )
#
# submit_from_classifier_calibrated(best_auc_c, best_auc_c_oof).to_csv(
#     os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_auc_c_calibrated.csv"), index=None
# )

# submit_from_sum_calibrated(best_auc_c, best_auc_c_oof).to_csv(
#     os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_auc_sum_calibrated.csv"), index=None
# )

# submit_from_sum_calibrated(best_loss, best_loss_oof).to_csv(
#     os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_loss_sum_calibrated.csv"), index=None
# )

# submit_from_sum_calibrated(
#     best_auc_b + best_auc_c + best_loss, best_auc_b_oof + best_auc_c_oof + best_loss_oof
# ).to_csv(os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_3_metrics_sum_calibrated.csv"), index=None)

# submit_from_classifier_calibrated(
#     best_auc_b + best_auc_c + best_loss, best_auc_b_oof + best_auc_c_oof + best_loss_oof
# ).to_csv(os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_3_metrics_c_calibrated.csv"), index=None)

# submit_from_average_classifier(
#     best_auc_b + best_auc_c + best_loss, best_auc_b_oof + best_auc_c_oof + best_loss_oof
# ).to_csv(os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_3_metrics_c_calibrated.csv"), index=None)

submit_from_median_classifier(best_auc_b + best_auc_c + best_loss).to_csv(
    os.path.join(output_dir, "ela_skresnext50_32x4d_fold0123_best_3_metrics_c_median.csv"), index=None
)
#
# for l, b, c in zip(best_loss_oof, best_auc_b_oof, best_auc_c_oof):
#     l = pd.read_csv(l)
#     b = pd.read_csv(b)
#     c = pd.read_csv(c)
#
#     y_true = l["true_modification_flag"].values.astype(int)
#
#     flag_x = []
#     type_x = []
#     for df in [l, b, c]:
#         df[OUTPUT_PRED_MODIFICATION_FLAG] = df[OUTPUT_PRED_MODIFICATION_FLAG].apply(sigmoid)
#         df[OUTPUT_PRED_MODIFICATION_TYPE] = df[OUTPUT_PRED_MODIFICATION_TYPE].apply(classifier_probas)
#         print("Flag", alaska_weighted_auc(y_true, df[OUTPUT_PRED_MODIFICATION_FLAG]))
#         print("Type", alaska_weighted_auc(y_true, df[OUTPUT_PRED_MODIFICATION_TYPE]))
#
#         flag_x.append(df[OUTPUT_PRED_MODIFICATION_FLAG].values)
#         type_x.append(df[OUTPUT_PRED_MODIFICATION_TYPE].values)
#
#     flag_x = np.dstack(flag_x)[0]
#     type_x = np.dstack(type_x)[0]
#
#     print("Flag (median)", alaska_weighted_auc(y_true, np.median(flag_x, axis=1)))
#     print("Type (median)", alaska_weighted_auc(y_true, np.median(type_x, axis=1)))


# for t, v in zip(best_auc_c, best_auc_c_oof):
#     print("Default")
#     calibrated(pd.read_csv(t), pd.read_csv(v))
#     print("calibrated_sum")
#     calibrated_raw(pd.read_csv(t), pd.read_csv(v))
