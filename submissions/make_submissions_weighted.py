import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings
from functools import partial

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd

# Classifiers
import scipy as sp
from scipy.optimize import Bounds
from scipy.special import softmax

from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import classifier_probas, sigmoid
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum

warnings.simplefilter("ignore")


def get_x_y(predictions):
    y = None
    X = []

    for p in predictions:
        p = pd.read_csv(p)
        if "true_modification_flag" in p:
            y = p["true_modification_flag"].values.astype(np.float32)

        X.append(np.expand_dims(p["pred_modification_flag"].apply(sigmoid).values, -1))
        X.append(np.expand_dims(p["pred_modification_type"].apply(classifier_probas).values, -1))

    X = np.column_stack(X).astype(np.float32)
    return X, y


def _auc_loss(coef, X, y):
    coef = softmax(coef)
    x_weighted = (np.expand_dims(coef, 0) * X).sum(axis=1)
    auc = alaska_weighted_auc(y, x_weighted)
    return 1 - auc


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        "May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        "May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        "May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        "May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
        #
        "Jun02_12_26_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        "Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        "Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        "Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
    ]

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")
    checksum = compute_checksum(test_predictions)

    X, y = get_x_y(holdout_predictions)
    print(X.shape, y.shape)

    X_public_lb, _ = get_x_y(test_predictions)
    print(X_public_lb.shape)

    loss_partial = partial(_auc_loss, X=X, y=y)
    initial_coef = np.ones(X.shape[1]) / X.shape[1]
    result = sp.optimize.minimize(
        loss_partial,
        initial_coef,
        bounds=Bounds(0, 1),
        method="nelder-mead",
        options={"maxiter": 5000, "disp": True, "gtol": 1e-10, "maxfun": 99999},
        tol=1e-6,
    )
    print(result)
    best_coef = softmax(result.x)
    print(best_coef)
    x_pred = (np.expand_dims(best_coef, 0) * X).sum(axis=1)
    auc = alaska_weighted_auc(y, x_pred)
    print(auc)

    x_test_pred = (np.expand_dims(best_coef, 0) * X_public_lb).sum(axis=1)

    submit_fname = os.path.join(output_dir, f"wmean_{np.mean(auc):.4f}_{checksum}.csv")

    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = x_test_pred
    df[["Id", "Label"]].to_csv(submit_fname, index=False)
    print("Saved submission to ", submit_fname)


if __name__ == "__main__":
    main()
