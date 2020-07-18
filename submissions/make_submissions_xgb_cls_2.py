import os

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from alaska2.dataset import METHOD_TO_INDEX
from alaska2.metric import alaska_weighted_auc


def get_x_y_for_stacking(fname, columns):
    df = pd.read_csv(fname)
    image_ids = df["NAME"].tolist()
    quality = df["QF"].tolist()
    target = df["CLASS"].values
    if (target == "?").any():
        y = None
    else:
        y = np.array(df["CLASS"].apply(lambda x: METHOD_TO_INDEX[x]).tolist())
        y = (y > 0).astype(int)

    x = []
    for col in columns:
        x.append(np.array(df[col].tolist()))

    x = np.column_stack(x)
    return x, y, quality, image_ids


def main():
    output_dir = os.path.dirname(__file__)

    columns = [
        "DCTR",
        "JRM",
        # "MixNet_xl_pc",
        # "MixNet_xl_pjm",
        # "MixNet_xl_pjuni",
        # "MixNet_xl_puerd",
        # "efn_b4_pc",
        # "efn_b4_pjm",
        # "efn_b4_pjuni",
        # "efn_b4_puerd",
        # "efn_b2_pc",
        # "efn_b2_pjm",
        # "efn_b2_pjuni",
        # "efn_b2_puerd",
        # "MixNet_s_pc",
        # "MixNet_s_pjm",
        # "MixNet_s_pjuni",
        # "MixNet_s_puerd",
        # "SRNet_pc",
        # "SRNet_pjm",
        # "SRNet_pjuni",
        # "SRNet_puerd",
        # "SRNet_noPC70_pc",
        # "SRNet_noPC70_pjm",
        # "SRNet_noPC70_pjuni",
        # "SRNet_noPC70_puerd",
        "efn_b4_mish_pc",
        "efn_b4_mish_pjm",
        "efn_b4_mish_pjuni",
        "efn_b4_mish_puerd",
        "efn_b5_mish_pc",
        "efn_b5_mish_pjm",
        "efn_b5_mish_pjuni",
        "efn_b5_mish_puerd",
        # "efn_b2_NR_mish_pc",
        # "efn_b2_NR_mish_pjm",
        # "efn_b2_NR_mish_pjuni",
        # "efn_b2_NR_mish_puerd",
        "MixNet_xl_mish_pc",
        "MixNet_xl_mish_pjm",
        "MixNet_xl_mish_pjuni",
        "MixNet_xl_mish_puerd",
        "efn_b6_NR_mish_pc",
        "efn_b6_NR_mish_pjm",
        "efn_b6_NR_mish_pjuni",
        "efn_b6_NR_mish_puerd",
        "SRNet_noPC70_mckpt_pc",
        "SRNet_noPC70_mckpt_pjm",
        "SRNet_noPC70_mckpt_pjuni",
        "SRNet_noPC70_mckpt_puerd",
    ]
    x, y, quality_h, image_ids = get_x_y_for_stacking("probabilities_zoo_holdout_0718.csv", columns)
    print(x.shape, y.shape)

    x_test, _, quality_t, image_ids_test = get_x_y_for_stacking("probabilities_zoo_lb_0718.csv", columns)
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

    checksum = "_".join(columns)
    submit_fname = os.path.join(output_dir, f"xgb_cls_2_{np.mean(cv_scores):.4f}_{checksum}.csv")
    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(submit_fname, index=False)
    print("Saved submission to ", submit_fname)


if __name__ == "__main__":
    main()
