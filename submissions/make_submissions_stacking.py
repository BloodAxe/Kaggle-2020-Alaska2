from collections import defaultdict

import torch
import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings

import matplotlib.pyplot as plt

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier  # <- Here is our boy
from pytorch_toolbelt.utils import fs
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier

# Classifiers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import classifier_probas, sigmoid, parse_array
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum, compute_checksum_v2

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
        # "A_May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        # "A_May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        # "A_May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        # "A_May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
        #
        "B_Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "B_Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "B_Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "B_Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "C_Jun02_12_26_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        "C_Jun24_22_00_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        "D_Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        "D_Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        # "E_Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "E_Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
        #
        "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
    ]

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")

    holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
    image_ids = np.array([fs.id_from_fname(x) for x in holdout_ds.images])

    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    x, y = get_x_y(holdout_predictions)
    print(x.shape, y.shape)

    x_test, _ = get_x_y(test_predictions)
    print(x_test.shape)

    x = np.column_stack([x, quality_h])
    x_test = np.column_stack([x_test, quality_t])

    group_kfold = GroupKFold(n_splits=5)

    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    auc_cv = []

    for fold, (train_index, valid_index) in enumerate(group_kfold.split(x, y, groups=image_ids)):
        x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]

        classifier1 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", LGBMClassifier())])
        classifier1.fit(x_train, y_train)

        classifier2 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", CatBoostClassifier())])
        classifier2.fit(x_train, y_train)

        classifier3 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", LogisticRegression())])
        classifier3.fit(x_train, y_train)

        classifier4 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", CalibratedClassifierCV())])
        classifier4.fit(x_train, y_train)

        classifier5 = Pipeline(
            steps=[("preprocessor", StandardScaler()), ("classifier", LinearDiscriminantAnalysis())]
        )
        classifier5.fit(x_train, y_train)

        sclf = StackingCVClassifier(
            classifiers=[classifier1, classifier2, classifier3, classifier4, classifier5],
            shuffle=False,
            use_probas=True,
            cv=5,
            meta_classifier=SVC(probability=True),
        )

        sclf.fit(x_train, y_train, image_ids[train_index])

        classifiers = {
            "LGBMClassifier": classifier1,
            "CatBoostClassifier": classifier2,
            "LogisticRegression": classifier3,
            "CalibratedClassifierCV": classifier4,
            "LinearDiscriminantAnalysis": classifier5,
            "Stack": sclf,
        }

        # Get results
        results = pd.DataFrame()
        for key in classifiers:
            # Make prediction on test set
            y_pred = classifiers[key].predict_proba(x_valid)[:, 1]

            # Save results in pandas dataframe object
            results[f"{key}"] = y_pred
            print(fold, key, alaska_weighted_auc(y_valid, y_pred))

        # Add the test set to the results object
        results["Target"] = y_valid

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

        for key, counter in zip(classifiers, range(len(sclf))):
            # Get predictions
            y_pred = results[key]

            # Get AUC
            auc = alaska_weighted_auc(y_valid, y_pred)
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
        plt.savefig(
            os.path.join(output_dir, f"Probability Distribution for each Classifier - Fold {fold}.png"), dpi=1080
        )

        # Making prediction on test set
        y_pred = sclf.predict_proba(x_valid)[:, 1]

        # Getting AUC
        auc = alaska_weighted_auc(y_valid, y_pred)
        auc_cv.append(auc)

        # Print results
        print(f"The AUC of the tuned Stacking classifier - fold {fold} is {auc:.4f}")

        df["Label_" + str(fold)] = sclf.predict_proba(x_test)[:, 1]

    df["Label"] = np.mean(
        [df["Label_0"].values, df["Label_1"].values, df["Label_2"].values, df["Label_3"].values, df["Label_4"].values]
    )
    df.to_csv(
        os.path.join(output_dir, f"stacking_{np.mean(auc_cv):.4f}_{compute_checksum_v2(test_predictions)}.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
