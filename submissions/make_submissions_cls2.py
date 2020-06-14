import pandas as pd
import numpy as np

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
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
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


def main():
    output_dir = os.path.dirname(__file__)

    if True:
        best_loss = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
        ]
        best_bauc = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
        ]
        best_cauc = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
        ]

        best_loss_oof = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
        ]
        best_bauc_oof = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_oof_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_oof_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best_oof_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best_oof_predictions.csv",
        ]
        best_cauc_oof = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
        ]

        best_loss_h = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
        ]
        best_bauc_h = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
        ]
        best_cauc_h = [
            "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
            "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
            "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
            "models/Jun10_08_49_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
        ]

        print("Holdout")
        y = None
        X = []

        for p in best_loss_h + best_bauc_h + best_cauc_h:
            p = pd.read_csv(p)
            y = p["true_modification_flag"]

            p["pred_modification_flag"] = p["pred_modification_flag"].apply(sigmoid)
            p["pred_modification_type"] = p["pred_modification_type"].apply(classifier_probas)

            X.append(p["pred_modification_flag"].values)
            X.append(p["pred_modification_type"].values)

        X = np.dstack(X).squeeze(0)
        print(X.shape, y.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.20, random_state=1000, shuffle=True
        )

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Initializing Support Vector classifier
        classifier1 = SVC(C=50, degree=3, tol=1e-4, gamma="auto", kernel="rbf", probability=True)

        # Initializing Multi-layer perceptron  classifier
        classifier2 = MLPClassifier(
            activation="relu",
            alpha=0.1,
            hidden_layer_sizes=(10, 10, 10),
            learning_rate="constant",
            max_iter=20000,
            random_state=1000,
        )

        # Initialing Nu Support Vector classifier
        classifier3 = NuSVC(degree=1, kernel="rbf", nu=0.25, probability=True)

        # Initializing Random Forest classifier
        classifier4 = RandomForestClassifier(
            n_estimators=128,
            criterion="gini",
            max_depth=10,
            max_features="auto",
            min_samples_leaf=0.005,
            min_samples_split=0.005,
            n_jobs=-1,
            random_state=1000,
        )

        sclf = StackingCVClassifier(
            classifiers=[classifier1, classifier2, classifier3, classifier4],
            shuffle=False,
            use_probas=True,
            cv=5,
            meta_classifier=SVC(probability=True),
        )

        classifiers = {"SVC": classifier1, "MLP": classifier2, "NuSVC": classifier3, "RF": classifier4, "Stack": sclf}

        # Train classifiers
        for key in classifiers:
            print("Fitting", key)
            # Get classifier
            classifier = classifiers[key]

            # Fit classifier
            classifier.fit(X_train, y_train)

            # Save fitted classifier
            classifiers[key] = classifier
            print("Fitting", key, "finished")

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
            textstr = f"AUC: {auc:.3f}"

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
            estimator=sclf, param_grid=params, cv=5, scoring=alaska_weighted_auc, verbose=10, n_jobs=-1
        )

        # Fit GridSearchCV
        grid.fit(X_train, y_train)

        # Making prediction on test set
        y_pred = grid.predict_proba(X_test)[:, 1]

        # Getting AUC
        auc = alaska_weighted_auc(y_test, y_pred)

        # Print results
        print(f"The AUC of the tuned Stacking classifier is {auc:.3f}")

        # Classifier labels
        classifier_labels = ["SVC", "MLP", "NuSVC", "RF"]

        # Get all unique combinations of classifier with a set size greater than or equal to 2
        combo_classifiers = []
        for ii in range(2, len(classifier_labels) + 1):
            for subset in itertools.combinations(classifier_labels, ii):
                combo_classifiers.append(subset)

        # Stack, tune, and evaluate stack of classifiers
        for combo in combo_classifiers:
            # Get labels of classifier to create a stack
            labels = list(combo)

            # Get classifiers
            classifier_combo = []
            for ii in range(len(labels)):
                label = classifier_labels[ii]
                classifier = classifiers[label]
                classifier_combo.append(classifier)

            # Initializing the StackingCV classifier
            sclf = StackingCVClassifier(
                classifiers=classifier_combo,
                shuffle=False,
                use_probas=True,
                cv=5,
                meta_classifier=SVC(probability=True),
                n_jobs=-1,
            )

            # Initialize GridSearchCV
            grid = GridSearchCV(
                estimator=sclf, param_grid=params, cv=5, scoring=alaska_weighted_auc, verbose=0, n_jobs=-1
            )

            # Fit GridSearchCV
            grid.fit(X_train, y_train)

            # Making prediction on test set
            y_pred = grid.predict_proba(X_test)[:, 1]

            # Getting AUC
            auc = alaska_weighted_auc(y_test, y_pred)

            # Print results
            print(f"AUC of stack {combo}: {auc:.3f}")


if __name__ == "__main__":
    main()
