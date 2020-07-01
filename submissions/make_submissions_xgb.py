import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pytorch_toolbelt.utils import fs
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import classifier_probas, sigmoid, parse_array
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum

# Classifiers

warnings.simplefilter("ignore")


def get_x_y(predictions):
    y = None
    X = []

    for p in predictions:
        p = pd.read_csv(p)
        if "true_modification_flag" in p:
            y = p["true_modification_flag"].values.astype(np.float32)

        X.append(np.expand_dims(p["pred_modification_flag"].values, -1))
        pred_modification_type = np.array(p["pred_modification_type"].apply(parse_array).tolist())
        X.append(pred_modification_type)

        X.append(np.expand_dims(p["pred_modification_flag"].apply(sigmoid).values, -1))
        X.append(np.expand_dims(p["pred_modification_type"].apply(classifier_probas).values, -1))

        if "pred_modification_type_tta" in p:
            X.append(p["pred_modification_type_tta"].apply(parse_array).tolist())

        if "pred_modification_flag_tta" in p:
            X.append(p["pred_modification_flag_tta"].apply(parse_array).tolist())

    X = np.column_stack(X).astype(np.float32)
    if y is not None:
        y = y.astype(int)
    return X, y


def xgb_weighted_auc(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y_true = dtrain.get_label()
    result = "wauc", alaska_weighted_auc(y_true.astype(int), predt)
    return result


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        # "May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        # "May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        # "May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        # "May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
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

    import torch.nn.functional as F

    holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
    image_ids = [fs.id_from_fname(x) for x in holdout_ds.images]
    print("Unique image ids", len(np.unique(image_ids)))
    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    X, y = get_x_y(holdout_predictions)
    print(X.shape, y.shape)

    x_test, _ = get_x_y(test_predictions)
    print(x_test.shape)

    if True:
        sc = StandardScaler()
        X = sc.fit_transform(X)
        x_test = sc.transform(x_test)

    X = np.column_stack([X, quality_h])
    x_test = np.column_stack([x_test, quality_t])
    test_dmatrix = xgb.DMatrix(x_test)

    group_kfold = GroupKFold(n_splits=5)
    cv_scores = []
    test_pred = None

    for train_index, valid_index in group_kfold.split(X, y, groups=image_ids):
        x_train, x_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
        print(np.bincount(y_train), np.bincount(y_valid))

        train_dmatrix = xgb.DMatrix(x_train.copy(), y_train.copy())
        valid_dmatrix = xgb.DMatrix(x_valid.copy(), y_valid.copy())

        params = {
            "base_score": 0.5,
            # "booster": "gblinear",
            "booster": "gbtree",
            "colsample_bylevel": 1,
            "colsample_bynode": 1,
            "colsample_bytree": 1,
            # "gamma": 1.0,
            "learning_rate": 0.01,
            "max_delta_step": 0,
            "objective": "binary:logistic",
            "eta": 0.1,
            "reg_lambda": 0,
            "subsample": 0.8,
            "scale_pos_weight": 1,
            "min_child_weight": 2,
            "max_depth": 6,
            "tree_method": "exact",
            "seed": 42,
            "alpha": 0.01,
            "lambda": 0.01,
            "n_estimators": 100,
            "gamma": 0.01,
            "disable_default_eval_metric": 1,
            # "eval_metric": "wauc",
        }

        xgb_model = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=150,
            verbose_eval=True,
            feval=xgb_weighted_auc,
            maximize=True,
            evals=[(valid_dmatrix, "validation")],
        )

        y_valid_pred = xgb_model.predict(valid_dmatrix)
        score = alaska_weighted_auc(y_valid, y_valid_pred)
        # score2 = alaska_weighted_auc(y_valid, torch.from_numpy(y_valid_pred).sigmoid().numpy())
        # print(score, score2, roc_auc_score(y_valid, y_valid_pred))
        cv_scores.append(score)

        if test_pred is not None:
            test_pred += xgb_model.predict(test_dmatrix)
        else:
            test_pred = xgb_model.predict(test_dmatrix)

    for s in cv_scores:
        print(s)
    print(np.mean(cv_scores), np.std(cv_scores))

    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(
        os.path.join(output_dir, f"{checksum}_xgb_cv_{np.mean(cv_scores):.4f}.csv"), index=False
    )


if __name__ == "__main__":
    main()
