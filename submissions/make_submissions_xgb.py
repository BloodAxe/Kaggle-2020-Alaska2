import os
import warnings
from typing import Tuple

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pytorch_toolbelt.utils import fs
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import classifier_probas, sigmoid, parse_array
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum, compute_checksum_v2
import torch.nn.functional as F


def get_x_y(predictions):
    y = None
    X = []

    for p in predictions:
        p = pd.read_csv(p)
        if "true_modification_flag" in p:
            y = p["true_modification_flag"].values.astype(np.float32)

        # X.append(np.expand_dims(p["pred_modification_flag"].values, -1))
        # pred_modification_type = np.array(p["pred_modification_type"].apply(parse_array).tolist())
        # X.append(pred_modification_type)

        X.append(np.expand_dims(p["pred_modification_flag"].apply(sigmoid).values, -1))
        X.append(np.expand_dims(p["pred_modification_type"].apply(classifier_probas).values, -1))
        X.append(
            np.expand_dims(
                p["pred_modification_type"].apply(classifier_probas).values
                * p["pred_modification_flag"].apply(sigmoid).values,
                -1,
            )
        )

        # if "pred_modification_type_tta" in p:
        #     X.append(p["pred_modification_type_tta"].apply(parse_array).tolist())
        #
        # if "pred_modification_flag_tta" in p:
        #     X.append(p["pred_modification_flag_tta"].apply(parse_array).tolist())

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
        # "A_May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        # "A_May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        # "A_May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        # "A_May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
        #
        # "B_Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "B_Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        # "B_Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        # "B_Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "C_Jun24_22_00_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        "D_Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        "D_Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        # "E_Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "E_Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
        #
        "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
        #
        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
    ]

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")
    fnames_for_checksum = [x + f"cauc" for x in experiments]
    checksum = compute_checksum_v2(fnames_for_checksum)

    holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
    image_ids = [fs.id_from_fname(x) for x in holdout_ds.images]
    print("Unique image ids", len(np.unique(image_ids)))
    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    x, y = get_x_y(holdout_predictions)
    print(x.shape, y.shape)

    x_test, _ = get_x_y(test_predictions)
    print(x_test.shape)

    if True:
        sc = StandardScaler()
        x = sc.fit_transform(x)
        x_test = sc.transform(x_test)

    if False:
        sc = PCA(n_components=16)
        x = sc.fit_transform(x)
        x_test = sc.transform(x_test)

    if True:
        x = np.column_stack([x, quality_h])
        x_test = np.column_stack([x_test, quality_t])

    test_dmatrix = xgb.DMatrix(x_test)

    group_kfold = GroupKFold(n_splits=5)
    cv_scores = []
    test_pred = None
    one_over_n = 1.0 / group_kfold.n_splits

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

    for fold_index, (train_index, valid_index) in enumerate(group_kfold.split(x, y, groups=image_ids)):
        x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]

        train_dmatrix = xgb.DMatrix(x_train.copy(), y_train.copy())
        valid_dmatrix = xgb.DMatrix(x_valid.copy(), y_valid.copy())



        xgb_model = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=1500,
            verbose_eval=True,
            feval=xgb_weighted_auc,
            maximize=True,
            evals=[(valid_dmatrix, "validation")],
        )

        y_valid_pred = xgb_model.predict_proba(valid_dmatrix)[:, 1]
        score = alaska_weighted_auc(y_valid, y_valid_pred)

        cv_scores.append(score)

        if test_pred is not None:
            test_pred += xgb_model.predict_proba(test_dmatrix)[:, 1] * one_over_n
        else:
            test_pred = xgb_model.predict_proba(test_dmatrix)[:, 1] * one_over_n

    for s in cv_scores:
        print(s)
    print(np.mean(cv_scores), np.std(cv_scores))

    submit_fname = os.path.join(output_dir, f"xgb_{np.mean(cv_scores):.4f}_{checksum}_.csv")
    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(submit_fname, index=False)
    print("Saved submission to ", submit_fname)


if __name__ == "__main__":
    main()
