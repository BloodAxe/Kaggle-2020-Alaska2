import os

from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import (
    make_classifier_predictions,
    make_classifier_predictions_calibrated,
    make_binary_predictions,
    make_binary_predictions_calibrated,
    blend_predictions_mean,
    blend_predictions_ranked,
)
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
    ]

    scoring_fn = alaska_weighted_auc
    # scoring_fn = shaky_wauc

    for metric in [
        # "loss",
        # "bauc",
        "cauc"
    ]:
        predictions_d4 = get_predictions_csv(experiments, metric, "holdout", "d4")
        oof_predictions_d4 = get_predictions_csv(experiments, metric, "holdout", "d4")
        test_predictions_d4 = get_predictions_csv(experiments, metric, "test", "d4")

        fnames_for_checksum = [x + f"{metric}" for x in experiments]

        bin_pred_d4 = make_binary_predictions(predictions_d4)
        y_true = bin_pred_d4[0].y_true_type.values

        bin_pred_d4_score = scoring_fn(y_true, blend_predictions_mean(bin_pred_d4).Label)

        cls_pred_d4 = make_classifier_predictions(predictions_d4)
        cls_pred_d4_score = scoring_fn(y_true, blend_predictions_mean(cls_pred_d4).Label)

        bin_pred_d4_cal = make_binary_predictions_calibrated(predictions_d4, oof_predictions_d4)
        bin_pred_d4_cal_score = scoring_fn(y_true, blend_predictions_mean(bin_pred_d4_cal).Label)

        cls_pred_d4_cal = make_classifier_predictions_calibrated(predictions_d4, oof_predictions_d4)
        cls_pred_d4_cal_score = scoring_fn(y_true, blend_predictions_mean(cls_pred_d4_cal).Label)

        prod_pred_d4_cal_score = scoring_fn(
            y_true, blend_predictions_mean(cls_pred_d4_cal).Label * blend_predictions_mean(bin_pred_d4_cal).Label
        )

        print(metric, "Bin NC", "d4", bin_pred_d4_score)
        print(metric, "Bin CL", "d4", cls_pred_d4_score)
        print(metric, "Cls NC", "d4", bin_pred_d4_cal_score)
        print(metric, "Cls CL", "d4", cls_pred_d4_cal_score)
        print(metric, "Prod  ", "d4", prod_pred_d4_cal_score)

        max_score = max(
            bin_pred_d4_score, cls_pred_d4_score, bin_pred_d4_cal_score, cls_pred_d4_cal_score, prod_pred_d4_cal_score
        )

        if bin_pred_d4_score == max_score:
            predictions = make_binary_predictions(test_predictions_d4)

            predictions = blend_predictions_mean(predictions)
            predictions.to_csv(
                os.path.join(output_dir, f"mean_{max_score:.4f}_bin_{compute_checksum_v2(fnames_for_checksum)}.csv"),
                index=False,
            )
        if bin_pred_d4_cal_score == max_score:
            predictions = make_binary_predictions_calibrated(test_predictions_d4, oof_predictions_d4)

            predictions = blend_predictions_mean(predictions)
            predictions.to_csv(
                os.path.join(
                    output_dir, f"mean_{max_score:.4f}_bin_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
                ),
                index=False,
            )
        if cls_pred_d4_score == max_score:
            predictions = make_classifier_predictions(test_predictions_d4)

            predictions = blend_predictions_mean(predictions)
            predictions.to_csv(
                os.path.join(output_dir, f"mean_{max_score:.4f}_cls_{compute_checksum_v2(fnames_for_checksum)}.csv"),
                index=False,
            )
        if cls_pred_d4_cal_score == max_score:
            predictions = make_classifier_predictions_calibrated(test_predictions_d4, oof_predictions_d4)

            predictions = blend_predictions_mean(predictions)
            predictions.to_csv(
                os.path.join(
                    output_dir, f"mean_{max_score:.4f}_cls_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
                ),
                index=False,
            )
        if prod_pred_d4_cal_score == max_score:
            cls_predictions = make_classifier_predictions_calibrated(test_predictions_d4, oof_predictions_d4)
            bin_predictions = make_binary_predictions_calibrated(test_predictions_d4, oof_predictions_d4)

            predictions1 = blend_predictions_mean(cls_predictions)
            predictions2 = blend_predictions_mean(bin_predictions)
            predictions = predictions1.copy()
            predictions.Label = predictions1.Label * predictions2.Label

            predictions.to_csv(
                os.path.join(
                    output_dir, f"mean_{max_score:.4f}_prod_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
                ),
                index=False,
            )


if __name__ == "__main__":
    main()
