import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings

import numpy as np

# Classifiers
from scipy.stats import rankdata
from tqdm import tqdm

from alaska2.metric import alaska_weighted_auc, shaky_wauc
from alaska2.submissions import blend_predictions_mean, make_binary_predictions, make_classifier_predictions
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2

# For reading, visualizing, and preprocessing data

warnings.simplefilter("ignore")

import itertools


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "H_Jul12_18_42_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
        "H_Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16",
        #
        "K_Jul17_17_09_nr_rgb_tf_efficientnet_b6_ns_mish_fold0_local_rank_0_fp16",
        "K_Jul18_16_41_nr_rgb_tf_efficientnet_b6_ns_mish_fold3_local_rank_0_fp16",
        #
        "J_Jul19_20_10_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
    ]

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")

    fnames_for_checksum = np.array(
        [x + "cauc_bin" for x in experiments]
        # + [x + "loss_bin" for x in experiments]
        + [x + "cauc_cls" for x in experiments]
        # + [x + "loss_cls" for x in experiments]
    )

    X = make_binary_predictions(holdout_predictions) + make_classifier_predictions(holdout_predictions)
    y_true = X[0].y_true_type.values
    X = np.array([x.Label.values for x in X])

    assert len(fnames_for_checksum) == X.shape[0]

    X_test = make_binary_predictions(test_predictions) + make_classifier_predictions(test_predictions)

    indices = np.arange(len(X))

    for r in range(2, 8):
        best_comb = None
        best_auc = 0
        combs = list(itertools.combinations(indices, r))

        for c in tqdm(combs, desc=f"{r}"):
            avg_preds = X[np.array(c)].mean(axis=0)
            score_averaging = alaska_weighted_auc(y_true, avg_preds)

            if score_averaging > best_auc:
                best_auc = score_averaging
                best_comb = c

        print(r, best_auc, best_comb)

        checksum = compute_checksum_v2(fnames_for_checksum[np.array(best_comb)])

        test_preds = [X_test[i] for i in best_comb]
        test_preds = blend_predictions_mean(test_preds)
        test_preds.to_csv(os.path.join(output_dir, f"cmb_mean_{best_auc:.4f}_{r}_{checksum}.csv"), index=False)

    for r in range(2, 8):
        best_comb = None
        best_auc = 0
        combs = list(itertools.combinations(indices, r))

        for c in tqdm(combs, desc=f"{r}"):
            rnk_preds = rankdata(X[np.array(c)], axis=1).mean(axis=0)
            score_averaging = alaska_weighted_auc(y_true, rnk_preds)

            if score_averaging > best_auc:
                best_auc = score_averaging
                best_comb = c

        print(r, best_auc, best_comb)

        checksum = compute_checksum_v2(fnames_for_checksum[np.array(best_comb)])

        test_preds = [X_test[i] for i in best_comb]
        test_preds = blend_predictions_mean(test_preds)
        test_preds.to_csv(os.path.join(output_dir, f"cmb_rank_{best_auc:.4f}_{r}_{checksum}.csv"), index=False)


if __name__ == "__main__":
    main()
