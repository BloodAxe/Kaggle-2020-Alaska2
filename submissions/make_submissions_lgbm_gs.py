import os

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
from alaska2.submissions import parse_classifier_probas, sigmoid, parse_array, parse_and_softmax, get_x_y_for_stacking
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2

import lightgbm as lgb


def wauc_metric(y_true, y_pred):
    wauc = alaska_weighted_auc(y_true, y_pred)
    return ("wauc", wauc, True)


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        # "H_Jul12_18_42_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
        #
        "K_Jul17_17_09_nr_rgb_tf_efficientnet_b6_ns_mish_fold0_local_rank_0_fp16",
        "J_Jul19_20_10_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
        "H_Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16",
        "K_Jul18_16_41_nr_rgb_tf_efficientnet_b6_ns_mish_fold3_local_rank_0_fp16"
        #
        #
    ]

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")
    checksum = compute_checksum_v2(experiments)

    holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
    image_ids = [fs.id_from_fname(x) for x in holdout_ds.images]

    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    with_logits = True

    x, y = get_x_y_for_stacking(holdout_predictions, with_logits=with_logits, tta_logits=with_logits)
    # Force target to be binary
    y = (y > 0).astype(int)
    print(x.shape, y.shape)

    x_test, _ = get_x_y_for_stacking(test_predictions, with_logits=with_logits, tta_logits=with_logits)
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

    group_kfold = GroupKFold(n_splits=5)

    params = {
        "boosting_type": ["gbdt", "dart", "rf", "goss"],
        "num_leaves": [16, 32, 64, 128],
        "reg_alpha": [0, 0.01, 0.1, 0.5],
        "reg_lambda": [0, 0.01, 0.1, 0.5],
        "learning_rate": [0.001, 0.01, 0.1, 0.5],
        "n_estimators": [32, 64, 126, 512],
        "max_depth": [2, 4, 8],
        "min_child_samples": [20, 40, 80, 100],
    }

    lgb_estimator = lgb.LGBMClassifier(objective="binary", silent=True)

    random_search = RandomizedSearchCV(
        lgb_estimator,
        param_distributions=params,
        scoring=make_scorer(alaska_weighted_auc, greater_is_better=True, needs_proba=True),
        n_jobs=3,
        n_iter=50,
        cv=group_kfold.split(x, y, groups=image_ids),
        verbose=2,
        random_state=42,
    )

    # Here we go
    random_search.fit(x, y)

    test_pred = random_search.predict_proba(x_test)[:, 1]
    print(test_pred)

    submit_fname = os.path.join(output_dir, f"lgbm_gs_{random_search.best_score_:.4f}_{checksum}.csv")
    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(submit_fname, index=False)

    print("\n All results:")
    print(random_search.cv_results_)
    print("\n Best estimator:")
    print(random_search.best_estimator_)
    print(random_search.best_score_)
    print("\n Best hyperparameters:")
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv("lgbm-random-grid-search-results-01.csv", index=False)

    # print(model.feature_importances_)


if __name__ == "__main__":
    main()
