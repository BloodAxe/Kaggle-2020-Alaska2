import torch
from pytorch_toolbelt.utils import fs
import os
# Used to ignore warnings generated from StackingCVClassifier
import warnings

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lazypredict.Supervised import LazyClassifier
from pytorch_toolbelt.utils import fs
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import (
    classifier_probas,
    sigmoid,
    parse_array,
)
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


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        # "A_May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        # "A_May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        # "A_May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        # "A_May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
        #
        "B_Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "B_Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "B_Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "B_Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "C_Jun02_12_26_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        "C_Jun24_22_00_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        "D_Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        "D_Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        # "E_Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "E_Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
        #
        "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
    ]

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")
    checksum = compute_checksum(test_predictions)

    holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
    image_ids = [fs.id_from_fname(x) for x in holdout_ds.images]

    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    x, y = get_x_y(holdout_predictions)
    print(x.shape, y.shape)

    x_test, _ = get_x_y(test_predictions)
    print(x_test.shape)

    if False:
        sc = StandardScaler()
        x = sc.fit_transform(x)
        x_test = sc.transform(x_test)

    x = np.column_stack([x, quality_h])
    x_test = np.column_stack([x_test, quality_t])

    group_kfold = GroupKFold(n_splits=5)
    cv_scores = []
    test_pred = None
    one_over_n = 1.0 / 5

    for i, (train_index, valid_index) in enumerate(group_kfold.split(x, y, groups=image_ids)):
        x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
        print(np.bincount(y_train), np.bincount(y_valid))

        clf = LazyClassifier(verbose=True, ignore_warnings=False, custom_metric=alaska_weighted_auc, predictions=True)
        models, predictions = clf.fit(x_train, x_valid, y_train, y_valid)
        print(models)

        models.to_csv(f"lazypredict_models_{i}.csv")
        predictions.to_csv(f"lazypredict_preds_{i}.csv")
        # if test_pred is not None:
        #     test_pred += cls.predict_proba(x_test)[:, 1] * one_over_n
        # else:
        #     test_pred = cls.predict_proba(x_test)[:, 1] * one_over_n

    # for s in cv_scores:
    #     print(s)
    # print(np.mean(cv_scores), np.std(cv_scores))

    # df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    # df["Label"] = test_pred
    # df[["Id", "Label"]].to_csv(
    #     os.path.join(output_dir, f"{checksum}_xgb_cls_cv_{np.mean(cv_scores):.4f}.csv"), index=False
    # )


if __name__ == "__main__":
    main()
