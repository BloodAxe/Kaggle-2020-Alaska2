import pandas as pd
import numpy as np
import torch

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from submissions.ela_skresnext50_32x4d import *
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
from sklearn.model_selection import train_test_split
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

warnings.simplefilter("ignore")


def get_x_y(predictions):
    y = None
    X = []

    for p in predictions:
        p = pd.read_csv(p)
        if "true_modification_flag" in p:
            y = p["true_modification_flag"].values

        X.append(np.expand_dims(p["pred_modification_flag"].values, -1))
        pred_modification_type = np.array(p["pred_modification_type"].apply(parse_array).tolist())
        X.append(pred_modification_type)

        X.append(np.expand_dims(p["pred_modification_flag"].apply(sigmoid).values, -1))
        X.append(np.expand_dims(p["pred_modification_type"].apply(classifier_probas).values, -1))

        if "pred_modification_type_tta" in p:
            X.append(p["pred_modification_type_tta"].apply(parse_array).tolist())

        if "pred_modification_flag_tta" in p:
            X.append(p["pred_modification_flag_tta"].apply(parse_array).tolist())

    X = np.column_stack(X)
    return X, y


def main():
    output_dir = os.path.dirname(__file__)

    if True:
        best_loss = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
            "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
        ]
        best_bauc = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
            "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
        ]
        best_cauc = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
            "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
        ]

        best_loss_h = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
            "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
        ]
        best_bauc_h = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
            "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
        ]
        best_cauc_h = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
            "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
        ]

        import torch.nn.functional as F

        holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
        quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float)

        test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
        quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float)

        X, y = get_x_y(as_d4_tta(best_loss_h + best_bauc_h + best_cauc_h))
        print(X.shape, y.shape)

        X_public_lb, _ = get_x_y(as_d4_tta(best_loss + best_bauc + best_cauc))
        print(X_public_lb.shape)

        X_train, X_test, y_train, y_test, quality_train, quality_test = train_test_split(
            X, y, quality_h, stratify=y, test_size=0.20, random_state=1000, shuffle=True
        )

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_public_lb = sc.transform(X_public_lb)

        X_train = np.column_stack([X_train, quality_train])
        X_test = np.column_stack([X_test, quality_test])
        X_public_lb = np.column_stack([X_public_lb, quality_t])

        # Initialize GridSearchCV
        parameters = {
            "n_estimators": [8, 16, 32, 64],
            "max_features": ["auto", "sqrt", "log2"],
            "max_depth": [2, 4, 8],
        }

        grid = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=parameters,
            cv=5,
            scoring=make_scorer(alaska_weighted_auc, greater_is_better=True, needs_proba=True),
            verbose=10,
            n_jobs=6,
        )

        # Fit GridSearchCV
        grid.fit(X_train, y_train)

        print(grid.best_params_)
        # Making prediction on test set
        y_pred = grid.predict_proba(X_test)[:, 1]

        # Getting AUC
        auc = alaska_weighted_auc(y_test, y_pred)

        # Print results
        print(f"The AUC of the tuned classifier on validation is {auc:.3f}")

        df = pd.read_csv(best_loss[0]).rename(columns={"image_id": "Id"})
        df["Label"] = grid.predict_proba(X_public_lb)[:, 1]
        df[["Id", "Label"]].to_csv(
            os.path.join(output_dir, f"rgb_tf_efficientnet_b6_ns_randomforest_gridsearch_{auc:.3f}.csv"), index=False
        )


if __name__ == "__main__":
    main()
