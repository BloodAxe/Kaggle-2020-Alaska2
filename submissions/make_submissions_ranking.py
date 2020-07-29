import os

from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import (
    make_classifier_predictions,
    make_classifier_predictions_calibrated,
    make_binary_predictions,
    make_binary_predictions_calibrated,
    blend_predictions_ranked,
    compute_checksum_v2,
)
from submissions.eval_tta import get_predictions_csv


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        # "A_May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        # "A_May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        # "A_May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        # "A_May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
        #
        # "B_Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "B_Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        # "B_Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        # "B_Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "C_Jun24_22_00_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        # "D_Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        # "D_Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        # "E_Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "E_Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
        #
        "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
        #
        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        # "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "H_Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16",
        "H_Jul12_18_42_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
    ]

    all_predictions = []
    labels = experiments
    scoring_fn = alaska_weighted_auc

    for metric in [
        # "loss",
        # "bauc",
        "cauc"
    ]:
        holdout_predictions_d4 = get_predictions_csv(experiments, metric, "holdout", "d4")
        oof_predictions_d4 = get_predictions_csv(experiments, metric, "oof", "d4")
        test_predictions_d4 = get_predictions_csv(experiments, metric, "test", "d4")

        fnames_for_checksum = [x + f"{metric}" for x in experiments]

        bin_pred_d4 = make_binary_predictions(holdout_predictions_d4)
        y_true = bin_pred_d4[0].y_true_type.values

        bin_pred_d4_score = scoring_fn(y_true, blend_predictions_ranked(bin_pred_d4).Label)

        cls_pred_d4 = make_classifier_predictions(holdout_predictions_d4)
        cls_pred_d4_score = scoring_fn(y_true, blend_predictions_ranked(cls_pred_d4).Label)

        prod_pred_d4_score = scoring_fn(
            y_true, blend_predictions_ranked(cls_pred_d4).Label * blend_predictions_ranked(bin_pred_d4).Label
        )

        if False:
            bin_pred_d4_cal = make_binary_predictions_calibrated(holdout_predictions_d4, oof_predictions_d4)
            bin_pred_d4_cal_score = scoring_fn(y_true, blend_predictions_ranked(bin_pred_d4_cal).Label)

            cls_pred_d4_cal = make_classifier_predictions_calibrated(holdout_predictions_d4, oof_predictions_d4)
            cls_pred_d4_cal_score = scoring_fn(y_true, blend_predictions_ranked(cls_pred_d4_cal).Label)

            prod_pred_d4_cal_score = scoring_fn(
                y_true,
                blend_predictions_ranked(cls_pred_d4_cal).Label * blend_predictions_ranked(bin_pred_d4_cal).Label,
            )
        else:
            bin_pred_d4_cal_score = 0
            cls_pred_d4_cal_score = 0
            prod_pred_d4_cal_score = 0

        print(metric, "Bin  NC", "d4", bin_pred_d4_score)
        print(metric, "Cls  NC", "d4", cls_pred_d4_score)
        print(metric, "Prod NC", "d4", prod_pred_d4_score)
        print(metric, "Bin  CL", "d4", bin_pred_d4_cal_score)
        print(metric, "Cls  CL", "d4", cls_pred_d4_cal_score)
        print(metric, "Prod CL", "d4", prod_pred_d4_cal_score)

        max_score = max(
            bin_pred_d4_score,
            cls_pred_d4_score,
            bin_pred_d4_cal_score,
            cls_pred_d4_cal_score,
            prod_pred_d4_score,
            prod_pred_d4_cal_score,
        )

        if bin_pred_d4_score == max_score:
            predictions = make_binary_predictions(test_predictions_d4)

            predictions = blend_predictions_ranked(predictions)
            predictions.to_csv(
                os.path.join(output_dir, f"rank_{max_score:.4f}_bin_{compute_checksum_v2(fnames_for_checksum)}.csv"),
                index=False,
            )
        if bin_pred_d4_cal_score == max_score:
            predictions = make_binary_predictions_calibrated(test_predictions_d4, oof_predictions_d4)

            predictions = blend_predictions_ranked(predictions)
            predictions.to_csv(
                os.path.join(
                    output_dir, f"rank_{max_score:.4f}_bin_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
                ),
                index=False,
            )
        if cls_pred_d4_score == max_score:
            predictions = make_classifier_predictions(test_predictions_d4)

            predictions = blend_predictions_ranked(predictions)
            predictions.to_csv(
                os.path.join(output_dir, f"rank_{max_score:.4f}_cls_{compute_checksum_v2(fnames_for_checksum)}.csv"),
                index=False,
            )
        if cls_pred_d4_cal_score == max_score:
            predictions = make_classifier_predictions_calibrated(test_predictions_d4, oof_predictions_d4)

            predictions = blend_predictions_ranked(predictions)
            predictions.to_csv(
                os.path.join(
                    output_dir, f"rank_{max_score:.4f}_cls_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
                ),
                index=False,
            )
        if prod_pred_d4_score == max_score:
            cls_predictions = make_classifier_predictions(test_predictions_d4)
            bin_predictions = make_binary_predictions(test_predictions_d4)

            predictions1 = blend_predictions_ranked(cls_predictions)
            predictions2 = blend_predictions_ranked(bin_predictions)
            predictions = predictions1.copy()
            predictions.Label = predictions1.Label * predictions2.Label

            predictions.to_csv(
                os.path.join(output_dir, f"rank_{max_score:.4f}_prod_{compute_checksum_v2(fnames_for_checksum)}.csv"),
                index=False,
            )
        if prod_pred_d4_cal_score == max_score:
            cls_predictions = make_classifier_predictions_calibrated(test_predictions_d4, oof_predictions_d4)
            bin_predictions = make_binary_predictions_calibrated(test_predictions_d4, oof_predictions_d4)

            predictions1 = blend_predictions_ranked(cls_predictions)
            predictions2 = blend_predictions_ranked(bin_predictions)
            predictions = predictions1.copy()
            predictions.Label = predictions1.Label * predictions2.Label

            predictions.to_csv(
                os.path.join(
                    output_dir, f"rank_{max_score:.4f}_prod_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
                ),
                index=False,
            )


if __name__ == "__main__":
    main()
