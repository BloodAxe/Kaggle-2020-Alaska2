import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alaska2.submissions import (
    make_classifier_predictions,
    blend_predictions_mean,
    parse_and_softmax,
)
from submissions.eval_tta import get_predictions_csv
# Used to ignore warnings generated from StackingCVClassifier
from submissions.make_submissions_averaging import evaluate_wauc_shakeup_using_bagging


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
        # "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
        #
        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
    ]

    test_predictions_d4 = get_predictions_csv(experiments, "cauc", "test", "d4")
    classes = []
    for x in test_predictions_d4:
        df = pd.read_csv(x)
        df = df.rename(columns={"image_id": "Id"})
        df["classes"] = df["pred_modification_type"].apply(parse_and_softmax)
        classes.append(df["classes"].tolist())

    classes = np.mean(classes, axis=0)


    print("Class distribution", np.bincount(classes.argmax(axis=1)))

    plt.figure()
    plt.hist(classes[:, 0], bins=100, alpha=0.25, label="Cover")
    plt.hist(classes[:, 1], bins=100, alpha=0.25, label="JMiPOD")
    plt.hist(classes[:, 2], bins=100, alpha=0.25, label="JUNIWARD")
    plt.hist(classes[:, 3], bins=100, alpha=0.25, label="UERD")
    plt.yscale("log")
    plt.legend()
    plt.show()

    holdout_predictions_d4 = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    holdout_predictions_d4 = make_classifier_predictions(holdout_predictions_d4)
    y_true_type = holdout_predictions_d4[0].y_true_type

    holdout_predictions_d4 = blend_predictions_mean(holdout_predictions_d4)
    scores = evaluate_wauc_shakeup_using_bagging(holdout_predictions_d4, y_true_type, 10000)

    plt.figure()
    plt.hist(scores, bins=100, alpha=0.5, label=f"{np.mean(scores):.5f} +- {np.std(scores):.6f}")
    plt.legend()
    plt.show()

    print(np.mean(scores), np.std(scores))


if __name__ == "__main__":
    main()
