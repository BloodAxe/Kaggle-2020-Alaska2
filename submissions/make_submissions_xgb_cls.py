import pandas as pd
import numpy as np
import torch
from pytorch_toolbelt.utils import fs
from xgboost import XGBClassifier

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from submissions.ela_skresnext50_32x4d import *
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum
from submissions.rgb_tf_efficientnet_b2_ns import *
from submissions.rgb_tf_efficientnet_b6_ns import *
from alaska2.submissions import (
    submit_from_classifier_calibrated,
    submit_from_average_classifier,
    blend_predictions_ranked,
    make_classifier_predictions,
    make_classifier_predictions_calibrated,
    make_binary_predictions,
    make_binary_predictions_calibrated,
    blend_predictions_mean,
    as_hv_tta,
    as_d4_tta,
    classifier_probas,
    sigmoid,
    parse_array,
)
from alaska2.metric import alaska_weighted_auc
import os
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Classifiers
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier  # <- Here is our boy

# Used to ignore warnings generated from StackingCVClassifier
import warnings
import xgboost as xgb

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
        "May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        "May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        "May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        "May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
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
    cv_scores = []
    test_pred = None

    for train_index, valid_index in group_kfold.split(X, y, groups=image_ids):
        x_train, x_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
        print(np.bincount(y_train), np.bincount(y_valid))

        cls = XGBClassifier(
            base_score=0.5,
            booster="gbtree",
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=0.8,
            gamma=1,
            gpu_id=-1,
            importance_type="gain",
            interaction_constraints="",
            learning_rate=0.02,
            max_delta_step=0,
            max_depth=5,
            min_child_weight=5,
            # missing=nan,
            monotone_constraints="()",
            n_estimators=600,
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
            verbosity=3,
        )

        cls.fit(x_train, y_train)

        y_valid_pred = cls.predict_proba(x_valid)[:, 1]
        score = alaska_weighted_auc(y_valid, y_valid_pred)
        cv_scores.append(score)

        if test_pred is not None:
            test_pred += cls.predict_proba(x_test)[:, 1]
        else:
            test_pred = cls.predict_proba(x_test)[:, 1]

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
