import os

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import fs
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
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

    params = {
        "min_child_weight": [1, 5, 10],
        "gamma": [1e-3, 1e-2, 1e-2, 0.5, 2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "max_depth": [2, 3, 4, 5, 6],
        "n_estimators": [16, 32, 64, 128, 256, 1000],
        "learning_rate": [0.001, 0.01, 0.05, 0.2, 1],
    }

    xgb = XGBClassifier(objective="binary:logistic", nthread=1)

    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=params,
        scoring=make_scorer(alaska_weighted_auc, greater_is_better=True, needs_proba=True),
        n_jobs=4,
        n_iter=100,
        cv=group_kfold.split(x, y, groups=image_ids),
        verbose=3,
        random_state=42,
    )

    # Here we go
    random_search.fit(x, y)

    print("\n All results:")
    print(random_search.cv_results_)
    print("\n Best estimator:")
    print(random_search.best_estimator_)
    print(random_search.best_score_)
    print("\n Best hyperparameters:")
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv("xgb-2-random-grid-search-results-01.csv", index=False)

    test_pred = random_search.predict_proba(x_test)[:, 1]

    checksum = "_".join(columns)
    submit_fname = os.path.join(output_dir, f"xgb_cls2_gs_{random_search.best_score_:.4f}_{checksum}_.csv")

    df = {}
    df["Label"] = test_pred
    df["Id"] = image_ids_test
    pd.DataFrame.from_dict(df).to_csv(submit_fname, index=False)
    print("Saved submission to ", submit_fname)

    import json

    with open(fs.change_extension(submit_fname, ".json"), "w") as f:
        json.dump(random_search.best_params_, f, indent=2)


if __name__ == "__main__":
    main()
