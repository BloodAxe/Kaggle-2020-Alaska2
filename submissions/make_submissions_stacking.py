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
        quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

        test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
        quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

        X, y = get_x_y(as_d4_tta(best_loss_h + best_bauc_h + best_cauc_h))
        print(X.shape, y.shape)

        X_public_lb, _ = get_x_y(as_d4_tta(best_loss + best_bauc + best_cauc))
        print(X_public_lb.shape)

        X_train, X_test, y_train, y_test, quality_train, quality_test = train_test_split(
            X, y, quality_h, stratify=y, test_size=0.20, random_state=1000, shuffle=True
        )

        sc = PCA(n_components=16)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_public_lb = sc.transform(X_public_lb)

        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)
        # X_public_lb = sc.transform(X_public_lb)

        X_train = np.column_stack([X_train, quality_train])
        X_test = np.column_stack([X_test, quality_test])
        X_public_lb = np.column_stack([X_public_lb, quality_t])

        # MLP
        # The AUC of the tuned MLP classifier is 0.936
        # Best params {'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (8,), 'learning_rate': 'adaptive', 'learning_rate_init': 0.0001, 'solver': 'adam'}
        # The AUC of the tuned MLP classifier is 0.935
        # Best params {'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': 16, 'learning_rate': 'adaptive', 'learning_rate_init': 0.0001, 'solver': 'adam'}
        classifier1 = MLPClassifier(
            activation="logistic",
            alpha=0.1,
            hidden_layer_sizes=(8,),
            learning_rate="adaptive",
            learning_rate_init=0.0001,
            solver="adam",
            max_iter=200000,
        )
        classifier1.fit(X_train, y_train)

        # RF
        # The AUC of the tuned RF classifier is 0.926
        # Best params {'max_depth': 6, 'max_features': 'auto', 'n_estimators': 64}
        # The AUC of the tuned RF classifier is 0.924
        # Best params {'max_depth': 6, 'max_features': 'auto', 'n_estimators': 64}
        classifier2 = RandomForestClassifier(max_depth=6, max_features="auto", n_estimators=64)
        classifier2.fit(X_train, y_train)

        # SVC
        # The AUC of the tuned SVC classifier is 0.937
        # Best params {'C': 50, 'degree': 2, 'kernel': 'linear'}
        classifier3 = SVC(probability=True, gamma="auto", C=50, degree=2, kernel="linear")
        classifier3.fit(X_train, y_train)

        sclf = StackingCVClassifier(
            classifiers=[classifier1, classifier2, classifier3],
            shuffle=False,
            use_probas=True,
            cv=5,
            meta_classifier=SVC(probability=True),
        )

        sclf.fit(X_train, y_train)

        classifiers = {"SVC": classifier1, "MLP": classifier2, "NuSVC": classifier3, "Stack": sclf}

        # Get results
        results = pd.DataFrame()
        for key in classifiers:
            # Make prediction on test set
            y_pred = classifiers[key].predict_proba(X_test)[:, 1]

            # Save results in pandas dataframe object
            results[f"{key}"] = y_pred

        # Add the test set to the results object
        results["Target"] = y_test

        # Probability Distributions Figure
        # Set graph style
        sns.set(font_scale=1)
        sns.set_style(
            {
                "axes.facecolor": "1.0",
                "axes.edgecolor": "0.85",
                "grid.color": "0.85",
                "grid.linestyle": "-",
                "axes.labelcolor": "0.4",
                "xtick.color": "0.4",
                "ytick.color": "0.4",
            }
        )

        # Plot
        f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols=5)

        for key, counter in zip(classifiers, range(5)):
            # Get predictions
            y_pred = results[key]

            # Get AUC
            auc = alaska_weighted_auc(y_test, y_pred)
            textstr = f"AUC: {auc:.4f}"

            # Plot false distribution
            false_pred = results[results["Target"] == 0]
            sns.distplot(
                false_pred[key],
                hist=True,
                kde=False,
                bins=int(25),
                color="red",
                hist_kws={"edgecolor": "black"},
                ax=ax[counter],
            )

            # Plot true distribution
            true_pred = results[results["Target"] == 1]
            sns.distplot(
                results[key],
                hist=True,
                kde=False,
                bins=int(25),
                color="green",
                hist_kws={"edgecolor": "black"},
                ax=ax[counter],
            )

            # These are matplotlib.patch.Patch properties
            props = dict(boxstyle="round", facecolor="white", alpha=0.5)

            # Place a text box in upper left in axes coords
            ax[counter].text(
                0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14, verticalalignment="top", bbox=props
            )

            # Set axis limits and labels
            ax[counter].set_title(f"{key} Distribution")
            ax[counter].set_xlim(0, 1)
            ax[counter].set_xlabel("Probability")

        # Tight layout
        plt.tight_layout()

        # Save Figure
        plt.savefig("Probability Distribution for each Classifier.png", dpi=1080)

        # Define parameter grid
        params = {
            "meta_classifier__kernel": ["linear", "rbf", "poly"],
            "meta_classifier__C": [1, 2],
            "meta_classifier__degree": [3, 4, 5],
            "meta_classifier__probability": [True],
        }

        # Initialize GridSearchCV
        grid = GridSearchCV(
            estimator=sclf,
            param_grid=params,
            cv=5,
            scoring=make_scorer(alaska_weighted_auc, greater_is_better=True, needs_proba=True),
            verbose=10,
            n_jobs=-1,
        )

        # Fit GridSearchCV
        grid.fit(X_train, y_train)

        # Making prediction on test set
        y_pred = grid.predict_proba(X_test)[:, 1]

        # Getting AUC
        auc = metrics.roc_auc_score(y_test, y_pred)

        # Print results
        print(f"The AUC of the tuned Stacking classifier is {auc:.4f}")


if __name__ == "__main__":
    main()
