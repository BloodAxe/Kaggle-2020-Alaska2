import os

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import fs
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.utils import parallel_backend
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import torch.nn.functional as F
import torch
from alaska2.dataset import METHOD_TO_INDEX
from alaska2.metric import alaska_weighted_auc

QUALITY_TO_INDEX = {75: 0, 90: 1, 95: 2}


def get_x_y_for_stacking(fname, columns):
    df = pd.read_csv(fname)
    image_ids = df["NAME"].tolist()
    quality = df["QF"].apply(lambda x: QUALITY_TO_INDEX[x]).tolist()
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

    checksum = "DCTR_JRM_B4_B5_B6_MixNet_XL_SRNET"
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
        quality_h = F.one_hot(torch.tensor(quality_h).long(), 3).numpy().astype(np.float32)
        quality_t = F.one_hot(torch.tensor(quality_t).long(), 3).numpy().astype(np.float32)

        x = np.column_stack([x, quality_h])
        x_test = np.column_stack([x_test, quality_t])

    group_kfold = GroupKFold(n_splits=5)


    bayes_cv_tuner = BayesSearchCV(
        estimator=XGBClassifier(objective="binary:logistic", gamma=1e-4, seed=42, nthread=1),
        search_spaces={
            "colsample_bylevel": (0.3, 1.0, "uniform"),
            "colsample_bytree": (0.3, 1.0, "uniform"),
            "colsample_bynode": (0.3, 1.0, "uniform"),
            "subsample": (0.3, 1.0, "uniform"),
            "learning_rate": (0.0001, 1.0, "log-uniform"),
            "eta": (0.01, 1.0, "uniform"),
            "max_depth": (2, 10),
            "alpha": (0.0001, 1.0, "log-uniform"),
            "lambda": (0.0001, 1.0, "log-uniform"),
            "n_estimators": (100, 300, "uniform"),
            "min_child_weight": (1, 10),
            "scale_pos_weight": (0.1, 1.0, "uniform"),
        },
        scoring=make_scorer(alaska_weighted_auc, greater_is_better=True, needs_proba=True),
        cv=5, #group_kfold.split(x, y, groups=image_ids),
        n_jobs=1,
        n_iter=200,
        verbose=2,
        refit=True,
        random_state=43,
    )

    # Here we go
    bayes_cv_tuner.fit(x, y, image_ids)

    print("\n All results:")
    print(random_search.cv_results_)
    print("\n Best estimator:")
    print(random_search.best_estimator_)
    print(random_search.best_score_)
    print("\n Best hyperparameters:")
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv("xgb-bayes-2-random-grid-search-results-01.csv", index=False)

    test_pred = bayes_cv_tuner.predict_proba(x_test)[:, 1]

    submit_fname = os.path.join(output_dir, f"xgb_bayes_cls2_gs_{random_search.best_score_:.4f}_{checksum}_.csv")

    df = {}
    df["Label"] = test_pred
    df["Id"] = image_ids_test
    pd.DataFrame.from_dict(df).to_csv(submit_fname, index=False)
    print("Saved submission to ", submit_fname)

    import json

    with open(fs.change_extension(submit_fname, ".json"), "w") as f:
        json.dump(random_search.best_params_, f, indent=2)

    print("Features importance")
    print(random_search.best_estimator_.feature_importances_)


if __name__ == "__main__":
    main()
