import os
import warnings

import pandas as pd
import torch
from pytorch_toolbelt.utils import logit, fs, to_numpy
from typing import List, Union, Tuple
import numpy as np
from scipy.stats import rankdata

from sklearn.calibration import CalibratedClassifierCV
from torch.nn import functional as F
from alaska2 import (
    OUTPUT_PRED_MODIFICATION_FLAG,
    alaska_weighted_auc,
    OUTPUT_PRED_MODIFICATION_TYPE,
    OUTPUT_PRED_EMBEDDING,
)
import re

__all__ = [
    "make_classifier_predictions",
    "make_classifier_predictions_calibrated",
    "make_binary_predictions",
    "make_binary_predictions_calibrated",
    "temperature_scaling",
    "sigmoid",
    "noop",
    "winsorize",
    "classifier_probas",
    "submit_from_average_binary",
    "submit_from_average_classifier",
    "submit_from_median_classifier",
    "calibrated",
    "blend_predictions_mean",
    "blend_predictions_ranked",
    "as_d4_tta",
    "as_hv_tta",
    "infer_fold",
    "compute_checksum_v2",
    "evaluate_wauc_shakeup_using_bagging",
]


def infer_fold(x):
    if "fold0" in x:
        return 0
    if "fold1" in x:
        return 1
    if "fold2" in x:
        return 2
    if "fold3" in x:
        return 3

    raise KeyError(x)


def temperature_scaling(x, t):
    x = torch.tensor(x)
    x_l = logit(x)
    x_s = torch.sigmoid(x_l * t)
    return float(x_s)


def sigmoid(x):
    return torch.sigmoid(torch.tensor(x)).item()


def parse_array(x):
    x = np.fromstring(x[1:-1], dtype=np.float32, sep=",")
    return x.tolist()


def parse_and_softmax(x):
    x = x.replace("[", "").replace("]", "").split(",")
    x = [float(i) for i in x]
    x = torch.tensor(x).softmax(dim=0)
    return to_numpy(x)


def classifier_probas(x):
    x = x.replace("[", "").replace("]", "").split(",")
    x = [float(i) for i in x]
    x = torch.tensor(x).softmax(dim=0)
    x = x[1:].sum()
    return float(x)


def noop(x):
    return x


def winsorize(x):
    import scipy.stats

    x_w = scipy.stats.mstats.winsorize(x, [0.05, 0.05])
    return x_w


def calibrated(test_predictions, oof_predictions, flag_transform=sigmoid, type_transform=classifier_probas):
    """
    Update test predictions w.r.t to calibration trained on OOF predictions
    :param test_predictions:
    :param oof_predictions:
    :return:
    """
    from sklearn.isotonic import IsotonicRegression as IR
    import matplotlib.pyplot as plt

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
    # print("Target", np.bincount(oof_predictions["true_modification_type"].values.astype(int)))

    if True:
        y_pred_raw = oof_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
        b_auc_before = alaska_weighted_auc(y_true, y_pred_raw)

        ir_flag = IR(out_of_bounds="clip", y_min=0, y_max=1)
        y_pred_cal = ir_flag.fit_transform(y_pred_raw, y_true)
        b_auc_after = alaska_weighted_auc(y_true, y_pred_cal)

        if b_auc_after > b_auc_before:
            test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = ir_flag.transform(
                test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
            )
        else:
            # test_predictions[OUTPUT_PRED_MODIFICATION_FLAG] = ir_flag.transform(
            #     test_predictions[OUTPUT_PRED_MODIFICATION_FLAG].values
            # )

            warnings.warn(f"Failed to train IR flag {b_auc_before} {b_auc_after}")

            plt.figure()
            plt.hist(y_pred_raw, alpha=0.5, bins=100, label=f"non-calibrated {b_auc_after}")
            plt.hist(y_pred_cal, alpha=0.5, bins=100, label=f"calibrated {b_auc_before}")
            plt.yscale("log")
            plt.legend()
            plt.show()

    if True:
        ir_type = IR(out_of_bounds="clip", y_min=0, y_max=1)
        y_pred_raw = oof_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
        c_auc_before = alaska_weighted_auc(y_true, y_pred_raw)
        y_pred_cal = ir_type.fit_transform(y_pred_raw, y_true)
        c_auc_after = alaska_weighted_auc(y_true, y_pred_cal)
        if c_auc_after > c_auc_before:
            test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = ir_type.transform(
                test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
            )

            # plt.figure()
            # plt.hist(y_pred_raw, alpha=0.5, bins=100, label=f"non-calibrated {c_auc_before}")
            # plt.hist(y_pred_cal, alpha=0.5, bins=100, label=f"calibrated {c_auc_after}")
            # plt.yscale("log")
            # plt.legend()
            # plt.show()
        else:
            # test_predictions[OUTPUT_PRED_MODIFICATION_TYPE] = ir_type.transform(
            #     test_predictions[OUTPUT_PRED_MODIFICATION_TYPE].values
            # )

            warnings.warn(f"Failed to train IR on type {c_auc_before} {c_auc_after}")

            # plt.figure()
            # plt.hist(y_pred_raw, alpha=0.5, bins=100, label=f"non-calibrated {c_auc_before}")
            # plt.hist(y_pred_cal, alpha=0.5, bins=100, label=f"calibrated {c_auc_after}")
            # plt.yscale("log")
            # plt.legend()
            # plt.show()

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
    submission["Label"] = sum([df["pred_modification_flag"].apply(sigmoid).values for df in preds_df]) / len(preds_df)
    return submission


def submit_from_average_classifier(preds: List[str]):
    assert isinstance(preds, list)
    preds_df = [pd.read_csv(x) for x in preds]

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
    submission["Label"] = sum([df["pred_modification_type"].apply(classifier_probas).values for df in preds_df]) / len(
        preds_df
    )
    return submission


def submit_from_median_classifier(test_predictions: List[str]):
    preds_df = [pd.read_csv(x) for x in test_predictions]

    p = np.stack([df["pred_modification_type"].apply(classifier_probas).values for df in preds_df])
    p = np.median(p, axis=0)

    submission = preds_df[0].copy().rename(columns={"image_id": "Id"})[["Id"]]
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


def make_binary_predictions(test_predictions: List[str]) -> List[pd.DataFrame]:
    preds_df = []
    for x in test_predictions:
        df = pd.read_csv(x).rename(columns={"image_id": "Id"})
        df["Label"] = df["pred_modification_flag"].apply(sigmoid)

        keys = ["Id", "Label"]
        if "true_modification_flag" in df:
            df["y_true"] = df["true_modification_flag"].astype(int)
            keys.append("y_true")

        if "true_modification_type" in df:
            df["y_true_type"] = df["true_modification_type"].astype(int)
            keys.append("y_true_type")

        preds_df.append(df[keys])

    return preds_df


def make_binary_predictions_calibrated(
    test_predictions: List[str], oof_predictions: List[str], print_results=False
) -> List[pd.DataFrame]:
    assert isinstance(test_predictions, list)
    assert isinstance(oof_predictions, list)
    assert len(oof_predictions) == len(test_predictions)

    preds_df = []
    for x, y in zip(test_predictions, oof_predictions):
        calibrated_test, scores = calibrated(pd.read_csv(x), pd.read_csv(y))
        if print_results:
            print(scores)

        calibrated_test["Id"] = calibrated_test["image_id"]
        calibrated_test["Label"] = calibrated_test["pred_modification_flag"]
        preds_df.append(calibrated_test[["Id", "Label"]])

    return preds_df


def make_classifier_predictions(test_predictions: List[str]) -> List[pd.DataFrame]:
    preds_df = []
    for x in test_predictions:
        df = pd.read_csv(x)
        df = df.rename(columns={"image_id": "Id"})
        df["Label"] = df["pred_modification_type"].apply(classifier_probas)

        keys = ["Id", "Label"]
        if "true_modification_flag" in df:
            df["y_true"] = df["true_modification_flag"].astype(int)
            keys.append("y_true")

        if "true_modification_type" in df:
            df["y_true_type"] = df["true_modification_type"].astype(int)
            keys.append("y_true_type")

        preds_df.append(df[keys])

    return preds_df


def make_product_predictions(test_predictions: List[str]) -> List[pd.DataFrame]:
    """
    Make predictions by multiplying binary and multiclass predictions
    """
    preds_df = []
    for x in test_predictions:
        df = pd.read_csv(x)
        df = df.rename(columns={"image_id": "Id"})
        df["Label"] = df["pred_modification_type"].apply(classifier_probas) * df["pred_modification_flag"].apply(
            sigmoid
        )

        keys = ["Id", "Label"]
        if "true_modification_flag" in df:
            df["y_true"] = df["true_modification_flag"].astype(int)
            keys.append("y_true")

        if "true_modification_type" in df:
            df["y_true_type"] = df["true_modification_type"].astype(int)
            keys.append("y_true_type")

        preds_df.append(df[keys])

    return preds_df


def make_classifier_predictions_calibrated(
    test_predictions: List[str], oof_predictions: List[str], print_results=False
) -> List[pd.DataFrame]:
    assert isinstance(test_predictions, list)
    assert isinstance(oof_predictions, list)
    assert len(oof_predictions) == len(test_predictions)

    preds_df = []
    for x, y in zip(test_predictions, oof_predictions):
        calibrated_test, scores = calibrated(pd.read_csv(x), pd.read_csv(y))
        if print_results:
            print(scores)

        calibrated_test["Id"] = calibrated_test["image_id"]
        calibrated_test["Label"] = calibrated_test["pred_modification_type"]

        keys = ["Id", "Label"]
        if "true_modification_flag" in calibrated_test:
            calibrated_test["y_true"] = calibrated_test["true_modification_flag"].astype(int)
            keys.append("y_true")

        preds_df.append(calibrated_test[keys])

    return preds_df


def get_x_y_for_stacking(
    predictions: List[str], with_logits=False, tta_logits=False, tta_probas=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get X and Y for 2nd level stacking
    """
    y = None
    X = []

    for p in predictions:
        p = pd.read_csv(p)

        if "true_modification_flag" in p:
            y = p["true_modification_flag"].values.astype(np.float32)

        X.append(np.expand_dims(p["pred_modification_flag"].apply(sigmoid).values, -1))
        X.append(np.expand_dims(p["pred_modification_type"].apply(classifier_probas).values, -1))
        X.append(
            np.expand_dims(
                p["pred_modification_type"].apply(classifier_probas).values
                * p["pred_modification_flag"].apply(sigmoid).values,
                -1,
            )
        )

        if with_logits:
            X.append(np.expand_dims(p["pred_modification_flag"].values, -1))
            X.append(np.array(p["pred_modification_type"].apply(parse_array).tolist()))

        if "pred_modification_type_tta" in p:

            def prase_tta_softmax(x):
                x = parse_array(x)
                x = torch.tensor(x)

                x = x.view((4, 8))
                x = x.softmax(dim=0)

                # x = x.view((8,4))
                # x = x.softmax(dim=1)
                x = x.view(-1)
                return x.tolist()

            if tta_logits:
                col = p["pred_modification_type_tta"].apply(parse_array)
                X.append(col.tolist())

            if tta_probas:
                col = p["pred_modification_type_tta"].apply(prase_tta_softmax)
                X.append(col.tolist())

        if "pred_modification_flag_tta" in p:
            if tta_logits:
                col = p["pred_modification_flag_tta"].apply(parse_array)
                X.append(col.tolist())
            if tta_probas:
                col = (
                    p["pred_modification_flag_tta"]
                    .apply(parse_array)
                    .apply(lambda x: torch.tensor(x).sigmoid().tolist())
                )
                X.append(col.tolist())

    X = np.column_stack(X).astype(np.float32)
    if y is not None:
        y = y.astype(int)
    return X, y


def get_x_y_embedding_for_stacking(predictions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get X and Y for 2nd level stacking
    """
    y = None
    X = []

    for p in predictions:
        p = pd.read_csv(p)

        if "true_modification_flag" in p:
            y = p["true_modification_flag"].values.astype(np.float32)

        X.append(np.array(p[OUTPUT_PRED_EMBEDDING].apply(parse_array).tolist()))

    X = np.column_stack(X).astype(np.float32)
    if y is not None:
        y = y.astype(int)
    return X, y


def evaluate_wauc_shakeup_using_bagging(oof_predictions: pd.DataFrame, y_true_type, n):
    wauc = []

    distribution = [3500, 500, 500, 500]

    oof_predictions = oof_predictions.copy()
    oof_predictions["y_true_type"] = y_true_type
    oof_predictions["y_true"] = y_true_type > 0

    cover = oof_predictions[oof_predictions["y_true_type"] == 0]
    juni = oof_predictions[oof_predictions["y_true_type"] == 1]
    jimi = oof_predictions[oof_predictions["y_true_type"] == 2]
    uerd = oof_predictions[oof_predictions["y_true_type"] == 3]

    for _ in range(n):
        bagging_df = pd.concat(
            [
                cover.sample(distribution[0]),
                juni.sample(distribution[1]),
                jimi.sample(distribution[2]),
                uerd.sample(distribution[3]),
            ]
        )
        auc = alaska_weighted_auc(bagging_df["y_true"], bagging_df["Label"])
        wauc.append(auc)

    return wauc


def compute_checksum_v2(fnames: List[str]):
    def sanitize_fname(x):
        x = fs.id_from_fname(x)
        x = (
            x.replace("fp16", "")
            .replace("fold", "f")
            .replace("local_rank_0", "")
            .replace("nr_rgb_tf_efficientnet_b6_ns", "")
            .replace("rgb_tf_efficientnet_b2_ns", "")
            .replace("rgb_tf_efficientnet_b3_ns", "")
            .replace("rgb_tf_efficientnet_b6_ns", "")
            .replace("rgb_tf_efficientnet_b7_ns", "")
        )
        x = re.sub(r"\w{3}\d{2}_\d{2}_\d{2}", "", x).replace("_", "")
        return x

    fnames = [sanitize_fname(x) for x in fnames]
    return "_".join(fnames)
