import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_toolbelt.utils import fs
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import classifier_probas, sigmoid, parse_array, parse_and_softmax
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2

import catboost as cat


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

        if "pred_modification_type_tta" in p:
            col = p["pred_modification_type_tta"].apply(parse_and_softmax)
            col_act = col.tolist()

            X.append(col_act)

        if "pred_modification_flag_tta" in p:
            col = p["pred_modification_flag_tta"].apply(parse_array)
            col_act = col.apply(lambda x: torch.tensor(x).sigmoid().tolist()).tolist()
            std = col.apply(lambda x: torch.tensor(x).sigmoid().std().item())

            X.append(col_act)
            X.append(std)

    X = np.column_stack(X).astype(np.float32)
    if y is not None:
        y = y.astype(int)
    return X, y


def wauc_metric(y_true, y_pred):
    wauc = alaska_weighted_auc(y_true, y_pred)
    return ("wauc", wauc, True)


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

    if True:
        x = np.column_stack([x, quality_h])
        x_test = np.column_stack([x_test, quality_t])

    group_kfold = GroupKFold(n_splits=2)

    params = {"learning_rate": [1e-3, 1e-2, 1e-1, 1], "depth": [4, 8, 16]}

    lgb_estimator = cat.CatBoostClassifier(iterations=1024, verbose=True)

    random_search = RandomizedSearchCV(
        lgb_estimator,
        param_distributions=params,
        n_iter=10,
        scoring=make_scorer(alaska_weighted_auc, greater_is_better=True, needs_proba=True),
        cv=group_kfold.split(x, y, groups=image_ids),
        verbose=3,
        random_state=42,
    )

    # Here we go
    random_search.fit(x, y)

    test_pred = random_search.predict_proba(x_test)[:, 1]
    print(test_pred)

    submit_fname = os.path.join(output_dir, f"catboost_gs_{random_search.best_score_:.4f}_{checksum}.csv")
    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(submit_fname, index=False)
    print("Saved predictions to ", submit_fname)

    print("\n All results:")
    print(random_search.cv_results_)
    print("\n Best estimator:")
    print(random_search.best_estimator_)
    print(random_search.best_score_)
    print("\n Best hyperparameters:")
    print(random_search.best_params_)

    # print(model.feature_importances_)


if __name__ == "__main__":
    main()
