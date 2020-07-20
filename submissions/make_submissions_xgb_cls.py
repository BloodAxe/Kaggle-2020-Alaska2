import os

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_toolbelt.utils import fs
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import get_x_y_for_stacking
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        # "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        # "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        # "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
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
    cv_scores = []
    test_pred = None
    one_over_n = 1.0 / group_kfold.n_splits

    for train_index, valid_index in group_kfold.split(x, y, groups=image_ids):
        x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]
        print(np.bincount(y_train), np.bincount(y_valid))

        cls = XGBClassifier(
            base_score=0.5,
            booster="gbtree",
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=0.6,
            gamma=0.5,
            gpu_id=-1,
            importance_type="gain",
            interaction_constraints="",
            learning_rate=0.01,
            max_delta_step=0,
            max_depth=3,
            min_child_weight=10,
            # missing=nan,
            monotone_constraints="()",
            n_estimators=1000,
            n_jobs=8,
            nthread=1,
            num_parallel_tree=1,
            objective="binary:logistic",
            random_state=0,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            silent=True,
            subsample=0.8,
            tree_method="exact",
            validate_parameters=1,
            verbosity=2,
        )

        cls.fit(x_train, y_train)

        y_valid_pred = cls.predict_proba(x_valid)[:, 1]
        score = alaska_weighted_auc(y_valid, y_valid_pred)
        cv_scores.append(score)

        if test_pred is not None:
            test_pred += cls.predict_proba(x_test)[:, 1] * one_over_n
        else:
            test_pred = cls.predict_proba(x_test)[:, 1] * one_over_n

    for s in cv_scores:
        print(s)
    print(np.mean(cv_scores), np.std(cv_scores))

    with_logits_sfx = "_with_logits" if with_logits else ""

    submit_fname = os.path.join(output_dir, f"xgb_cls_{np.mean(cv_scores):.4f}_{checksum}{with_logits_sfx}.csv")
    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(submit_fname, index=False)
    print("Saved submission to ", submit_fname)


if __name__ == "__main__":
    main()
