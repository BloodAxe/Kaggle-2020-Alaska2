import os

# For reading, visualizing, and preprocessing data
import numpy as np

from alaska2.submissions import get_x_y_for_stacking
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        #
        # "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        # "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "H_Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16",
        "H_Jul12_18_42_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
        #
        "K_Jul17_17_09_nr_rgb_tf_efficientnet_b6_ns_mish_fold0_local_rank_0_fp16",
    ]

    checksum = compute_checksum_v2(experiments)

    if False:
        train_predictions = get_predictions_csv(experiments, "cauc", "train", tta="d4", need_embedding=True)
        x, y = get_x_y_for_stacking(
            train_predictions,
            with_embeddings=True,
            with_logits=False,
            with_probas=False,
            tta_probas=False,
            tta_logits=False,
        )
        print("Train", x.shape, y.shape)
        np.save(f"embeddings_x_train_{checksum}.npy", x)
        np.save(f"embeddings_y_train_{checksum}.npy", y)
        del x, y, train_predictions

    if False:
        test_predictions = get_predictions_csv(experiments, "cauc", "test", tta="d4", need_embedding=True)
        x_test, _ = get_x_y_for_stacking(
            test_predictions,
            with_embeddings=True,
            with_logits=False,
            with_probas=False,
            tta_probas=False,
            tta_logits=False,
        )
        print("Test", x_test.shape)
        np.save(f"embeddings_x_test_{checksum}.npy", x_test)
        del x_test, test_predictions

    if True:
        holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", tta="d4", need_embedding=True)
        x_hld, y_hld = get_x_y_for_stacking(
            holdout_predictions,
            with_embeddings=True,
            with_logits=False,
            with_probas=False,
            tta_probas=False,
            tta_logits=False,
        )
        print("Holdout", x_hld.shape)
        np.save(f"embeddings_x_holdout_{checksum}.npy", x_hld)
        np.save(f"embeddings_y_holdout_{checksum}.npy", y_hld)
        del x_hld, y_hld, holdout_predictions


if __name__ == "__main__":
    main()
