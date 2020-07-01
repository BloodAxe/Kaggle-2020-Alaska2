import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.utils import fs
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.submissions import classifier_probas, sigmoid, parse_array
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, make_scorer
from alaska2.metric import alaska_weighted_auc

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

    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    X, y = get_x_y(holdout_predictions)
    print(X.shape, y.shape)

    x_test, _ = get_x_y(test_predictions)
    print(x_test.shape)

    if False:
        sc = StandardScaler()
        X = sc.fit_transform(X)
        x_test = sc.transform(x_test)

    X = np.column_stack([X, quality_h])
    x_test = np.column_stack([x_test, quality_t])

    group_kfold = GroupKFold(n_splits=5)

    params = {
        "min_child_weight": [1, 5, 10],
        "gamma": [0.1, 0.5, 1, 1.5, 2, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5],
        "n_estimators": [100, 200, 600],
        "learning_rate": [0.2, 0.5, 1.0],
    }

    xgb = XGBClassifier(objective="binary:logistic", nthread=1)

    param_comb = 5

    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=params,
        n_iter=param_comb,
        scoring=make_scorer(alaska_weighted_auc, greater_is_better=True, needs_proba=True),
        n_jobs=4,
        cv=group_kfold.split(X, y, groups=image_ids),
        verbose=3,
        random_state=1001,
    )

    # Here we go
    random_search.fit(X, y)

    print("\n All results:")
    print(random_search.cv_results_)
    print("\n Best estimator:")
    print(random_search.best_estimator_)
    print(random_search.best_score_)
    print("\n Best hyperparameters:")
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv("xgb-random-grid-search-results-01.csv", index=False)

    # print(model.feature_importances_)

    test_pred = random_search.predict_proba(x_test)[:, 1]
    print(test_pred)

    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(
        os.path.join(output_dir, f"{checksum}_xgb_gs_cv_{random_search.best_score_:.4f}.csv"), index=False
    )


if __name__ == "__main__":
    main()
