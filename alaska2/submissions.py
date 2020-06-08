import os
import pandas as pd
import torch
from pytorch_toolbelt.utils import logit, fs
from typing import List, Union
import numpy as np
from scipy.stats import rankdata

from sklearn.calibration import CalibratedClassifierCV
from torch.nn import functional as F
from alaska2 import OUTPUT_PRED_MODIFICATION_FLAG, alaska_weighted_auc, OUTPUT_PRED_MODIFICATION_TYPE

__all__ = [
    "make_classifier_predictions",
    "make_classifier_predictions_calibrated",
    "make_binary_predictions_calibrated",
    "temperature_scaling",
    "sigmoid",
    "noop",
    "winsorize",
    "classifier_probas",
    "stringify_image_id",
    "submit_from_average_binary",
    "submit_from_average_classifier",
    "submit_from_median_classifier",
    "calibrated",
    "blend_predictions_mean",
    "blend_predictions_ranked",
    "as_d4_tta",
    "as_hv_tta",
]


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


def noop(x):
    return x


def winsorize(x):
    import scipy.stats

    x_w = scipy.stats.mstats.winsorize(x, [0.05, 0.05])
    return x_w


def calibrated(test_predictions, oof_predictions, flag_transform=noop, type_transform=classifier_probas):
    """
    Update test predictions w.r.t to calibration trained on OOF predictions
    :param test_predictions:
    :param oof_predictions:
    :return:
    """
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
        b_auc_before = alaska_weighted_auc(y_true, oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values)
        b_auc_after = alaska_weighted_auc(y_true, flag_calibrated)
        print("Flag", b_auc_before, b_auc_after, (b_auc_after - b_auc_before))
        test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = ir_flag.transform(
            test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
        )

    if True:
        ir_type = IR(out_of_bounds="clip")
        ir_type.fit(oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values, y_true)
        type_calibrated = ir_type.transform(oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values)
        c_auc_before = alaska_weighted_auc(y_true, oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values)
        c_auc_after = alaska_weighted_auc(y_true, type_calibrated)
        print("Type", c_auc_before, c_auc_after, c_auc_after - c_auc_before)
        test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = ir_type.transform(
            test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
        )
    results = {
        "b_auc_before": b_auc_before,
        "b_auc_after": b_auc_after,
        "c_auc_before": c_auc_before,
        "c_auc_after": c_auc_after,
    }
    return test_predictions, results


def submit_from_average_binary(preds: List[str]):
    preds_df = [pd.read_csv(x) for x in preds]

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_flag"].apply(sigmoid).values for df in preds_df]) / len(preds_df)
    return submission


def submit_from_average_classifier(preds: List[str]):
    assert isinstance(preds, list)
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
    assert isinstance(test_predictions, list)
    assert isinstance(oof_predictions, list)
    assert len(oof_predictions) == len(test_predictions)

    preds_df = []
    for x, y in zip(test_predictions, oof_predictions):
        calibrated_test, scores = calibrated(pd.read_csv(x), pd.read_csv(y))
        print(scores)
        preds_df.append(calibrated_test)

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_flag"].values for df in preds_df]) / len(preds_df)
    return submission


def submit_from_classifier_calibrated(test_predictions: List[str], oof_predictions: List[str]):
    assert isinstance(test_predictions, list)
    assert isinstance(oof_predictions, list)
    assert len(oof_predictions) == len(test_predictions)

    preds_df = []
    for x, y in zip(test_predictions, oof_predictions):
        calibrated_test, scores = calibrated(pd.read_csv(x), pd.read_csv(y))
        print(scores)
        preds_df.append(calibrated_test)

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Id"] = submission["Id"].apply(stringify_image_id)
    submission["Label"] = sum([df["pred_modification_type"].values for df in preds_df]) / len(preds_df)
    return submission


def blend_predictions_mean(predictions: List[pd.DataFrame], winsorized=False):
    if winsorized:
        op = winsorize
    else:
        op = noop

    df = [pd.read_csv(x) if isinstance(x, str) else x for x in predictions]
    return pd.DataFrame.from_dict({"Id": df[0].Id.tolist(), "Label": sum([op(x.Label.values) for x in df]) / len(df)})


def blend_predictions_ranked(predictions: List[Union[str, pd.DataFrame]]):
    df = [pd.read_csv(x) if isinstance(x, str) else x for x in predictions]
    return pd.DataFrame.from_dict(
        {"Id": df[0].Id.tolist(), "Label": sum([rankdata(x.Label.values) for x in df]) / len(df)}
    )


def as_hv_tta(predictions):
    return [fs.change_extension(x, "_flip_hv_tta.csv") for x in predictions]


def as_d4_tta(predictions):
    return [fs.change_extension(x, "_d4_tta.csv") for x in predictions]


def make_binary_predictions_calibrated(test_predictions: List[str], oof_predictions: List[str]) -> List[pd.DataFrame]:
    assert isinstance(test_predictions, list)
    assert isinstance(oof_predictions, list)
    assert len(oof_predictions) == len(test_predictions)

    preds_df = []
    for x, y in zip(test_predictions, oof_predictions):
        calibrated_test, scores = calibrated(pd.read_csv(x), pd.read_csv(y))
        print(scores)

        calibrated_test["Id"] = calibrated_test["image_id"].apply(stringify_image_id)
        calibrated_test["Label"] = calibrated_test["pred_modification_flag"]
        preds_df.append(calibrated_test[["Id", "Label"]])

    return preds_df


def make_classifier_predictions(test_predictions: List[str]) -> List[pd.DataFrame]:
    preds_df = []
    for x in test_predictions:
        df = pd.read_csv(x).rename(columns={"image_id": "Id"})
        df["Id"] = df["Id"].apply(stringify_image_id)
        df["Label"] = df["pred_modification_type"].apply(classifier_probas)

        keys = ["Id", "Label"]
        if "true_modification_flag" in df:
            df["y_true"] = df["true_modification_flag"].astype(int)
            keys.append("y_true")

        preds_df.append(df[keys])

    return preds_df


def make_classifier_predictions_calibrated(
    test_predictions: List[str], oof_predictions: List[str]
) -> List[pd.DataFrame]:
    assert isinstance(test_predictions, list)
    assert isinstance(oof_predictions, list)
    assert len(oof_predictions) == len(test_predictions)

    preds_df = []
    for x, y in zip(test_predictions, oof_predictions):
        calibrated_test, scores = calibrated(pd.read_csv(x), pd.read_csv(y))
        print(scores)

        calibrated_test["Id"] = calibrated_test["image_id"].apply(stringify_image_id)
        calibrated_test["Label"] = calibrated_test["pred_modification_type"]

        keys = ["Id", "Label"]
        if "true_modification_flag" in calibrated_test:
            calibrated_test["y_true"] = calibrated_test["true_modification_flag"].astype(int)
            keys.append("y_true")

        preds_df.append(calibrated_test[keys])

    return preds_df
