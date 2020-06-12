import pandas as pd
import numpy as np
from mlxtend.classifier import StackingCVClassifier

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
)
from alaska2.metric import alaska_weighted_auc
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

import matplotlib.pyplot as plt


def main():
    output_dir = os.path.dirname(__file__)

    best_loss = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
    ]
    best_bauc = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
    ]
    best_cauc = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
    ]

    best_loss_oof = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
    ]
    best_bauc_oof = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_oof_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_oof_predictions.csv",
    ]
    best_cauc_oof = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
    ]

    best_loss_h = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
    ]
    best_bauc_h = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
    ]
    best_cauc_h = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
    ]

    sclf = StackingCVClassifier(classifiers=[classifier1, classifier2, classifier3, classifier4],
                                shuffle=False,
                                use_probas=True,
                                cv=5,
                                meta_classifier=SVC(probability=True))

    for p in best_loss_oof + best_bauc_oof + best_cauc_oof:
        print(p)
        p = pd.read_csv(p)
        y_true = p["true_modification_flag"].values
        y_pred = p["pred_modification_type"].apply(classifier_probas).values

        print(
            alaska_weighted_auc(y_true, p["pred_modification_flag"]),
            alaska_weighted_auc(y_true, p["pred_modification_type"].apply(classifier_probas)),
        )

        # https://towardsdatascience.com/stacking-classifiers-for-higher-predictive-performance-566f963e4840
        # Initializing Support Vector classifier
        classifier1 = SVC(C=50, degree=1, gamma="auto", kernel="rbf", probability=True)

        # Initializing Multi-layer perceptron  classifier
        classifier2 = MLPClassifier(
            activation="relu",
            alpha=0.1,
            hidden_layer_sizes=(10, 10, 10),
            learning_rate="constant",
            max_iter=2000,
            random_state=1000,
        )

        # Initialing Nu Support Vector classifier
        classifier3 = NuSVC(degree=1, kernel="rbf", nu=0.25, probability=True)

        # Initializing Random Forest classifier
        classifier4 = RandomForestClassifier(
            n_estimators=500,
            criterion="gini",
            max_depth=10,
            max_features="auto",
            min_samples_leaf=0.005,
            min_samples_split=0.005,
            n_jobs=-1,
            random_state=1000,
        )

        # p1 = blend_predictions_mean(p1)
        # print("best_loss_h", alaska_weighted_auc(y_true_holdout, p1["Label"]))
        #
        p1 = make_classifier_predictions([best_loss_h[0]])
        y_true_holdout = p1[0]["y_true"]
        print("best_loss_h", alaska_weighted_auc(y_true_holdout, blend_predictions_mean(p1)["Label"]))

        p3 = make_classifier_predictions(best_cauc_h)
        print("best_cauc_h", alaska_weighted_auc(y_true_holdout, blend_predictions_mean(p3)["Label"]))
        #

        p_mean = blend_predictions_mean(p1 + p3)
        print("Averaged", alaska_weighted_auc(y_true_holdout, p_mean["Label"]))

        p1 = make_classifier_predictions([best_loss[0]])
        p3 = make_classifier_predictions(best_cauc)
        p_mean = blend_predictions_mean(p1 + p3)
        p_mean.to_csv(
            os.path.join(output_dir, "rgb_tf_efficientnet_b6_ns_fold01_blend_mean_9278_holdout.csv"), index=False
        )

        p1 = make_classifier_predictions(best_loss)
        p3 = make_classifier_predictions(best_cauc)
        p_mean = blend_predictions_mean(p1 + p3)
        p_mean.to_csv(
            os.path.join(output_dir, "rgb_tf_efficientnet_b6_ns_fold01_blend_mean_9274_holdout.csv"), index=False
        )

        predictions = make_classifier_predictions(
            best_loss
            + best_cauc
            + as_hv_tta(ela_skresnext50_32x4d_best_loss)
            + as_hv_tta(rgb_tf_efficientnet_b6_ns_best_loss)
            + as_hv_tta(rgb_tf_efficientnet_b2_ns_best_auc_c)
        )
        blend_predictions_mean(predictions).to_csv(os.path.join(output_dir, "ranked.csv"), index=False)


if __name__ == "__main__":
    main()
