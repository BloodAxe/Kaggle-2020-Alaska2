import os
import numpy as np

from alaska2.metric import alaska_weighted_auc, shaky_wauc, shaky_wauc_public
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
        "B_Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "B_Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "B_Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "B_Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",

        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
    ]

    for metric in [
        # "loss",
        # "bauc",
        "cauc"
    ]:
        holdout_predictions_d4 = get_predictions_csv(experiments, metric, "holdout", "d4")
        oof_predictions_d4 = get_predictions_csv(experiments, metric, "oof", "d4")
        test_predictions_d4 = get_predictions_csv(experiments, metric, "test", "d4")

        hld_bin_pred_d4 = make_binary_predictions(holdout_predictions_d4)
        hld_y_true = hld_bin_pred_d4[0].y_true_type.values

        oof_bin_pred_d4 = make_binary_predictions(oof_predictions_d4)

        hld_cls_pred_d4 = make_classifier_predictions(holdout_predictions_d4)
        oof_cls_pred_d4 = make_classifier_predictions(oof_predictions_d4)

        bin_pred_d4_cal = make_binary_predictions_calibrated(holdout_predictions_d4, oof_predictions_d4)
        cls_pred_d4_cal = make_classifier_predictions_calibrated(holdout_predictions_d4, oof_predictions_d4)

        print("   ", "      ", "  ", "   OOF", "     OOF 5K", "     OOF 1K", "        HLD", "     HLD 5K", "     HLD 1K")
        print(
            metric,
            "Bin NC",
            "{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
                np.mean([alaska_weighted_auc(x.y_true_type, x.Label) for x in oof_bin_pred_d4]),
                np.mean([shaky_wauc(x.y_true_type, x.Label) for x in oof_bin_pred_d4]),
                np.mean([shaky_wauc_public(x.y_true_type, x.Label) for x in oof_bin_pred_d4]),
                alaska_weighted_auc(hld_y_true, blend_predictions_mean(hld_bin_pred_d4).Label),
                shaky_wauc(hld_y_true, blend_predictions_mean(hld_bin_pred_d4).Label),
                shaky_wauc_public(hld_y_true, blend_predictions_mean(hld_bin_pred_d4).Label),
            ),
        )

        print(
            metric,
            "Cls NC",
            "{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
                np.mean([alaska_weighted_auc(x.y_true_type, x.Label) for x in oof_cls_pred_d4]),
                np.mean([shaky_wauc(x.y_true_type, x.Label) for x in oof_cls_pred_d4]),
                np.mean([shaky_wauc_public(x.y_true_type, x.Label) for x in oof_cls_pred_d4]),
                alaska_weighted_auc(hld_y_true, blend_predictions_mean(hld_cls_pred_d4).Label),
                shaky_wauc(hld_y_true, blend_predictions_mean(hld_cls_pred_d4).Label),
                shaky_wauc_public(hld_y_true, blend_predictions_mean(hld_cls_pred_d4).Label),
            ),
        )

        print(
            metric,
            "Bin CL",
            "                                    {:.6f}\t{:.6f}\t{:.6f}".format(
                alaska_weighted_auc(hld_y_true, blend_predictions_mean(bin_pred_d4_cal).Label),
                shaky_wauc(hld_y_true, blend_predictions_mean(bin_pred_d4_cal).Label),
                shaky_wauc_public(hld_y_true, blend_predictions_mean(bin_pred_d4_cal).Label),
            ),
        )
        print(
            metric,
            "Cls CL",
            "                                    {:.6f}\t{:.6f}\t{:.6f}".format(
                alaska_weighted_auc(hld_y_true, blend_predictions_mean(cls_pred_d4_cal).Label),
                shaky_wauc(hld_y_true, blend_predictions_mean(cls_pred_d4_cal).Label),
                shaky_wauc_public(hld_y_true, blend_predictions_mean(cls_pred_d4_cal).Label),
            ),
        )
        print(
            metric,
            "Prd NC",
            "{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
                np.mean(
                    [
                        alaska_weighted_auc(x.y_true_type, x.Label * y.Label)
                        for (x, y) in zip(oof_bin_pred_d4, oof_cls_pred_d4)
                    ]
                ),
                np.mean(
                    [shaky_wauc(x.y_true_type, x.Label * y.Label) for (x, y) in zip(oof_bin_pred_d4, oof_cls_pred_d4)]
                ),
                np.mean(
                    [
                        shaky_wauc_public(x.y_true_type, x.Label * y.Label)
                        for (x, y) in zip(oof_bin_pred_d4, oof_cls_pred_d4)
                    ]
                ),
                alaska_weighted_auc(
                    hld_y_true,
                    blend_predictions_mean(bin_pred_d4_cal).Label * blend_predictions_mean(cls_pred_d4_cal).Label,
                ),
                shaky_wauc(
                    hld_y_true,
                    blend_predictions_mean(bin_pred_d4_cal).Label * blend_predictions_mean(cls_pred_d4_cal).Label,
                ),
                shaky_wauc_public(
                    hld_y_true,
                    blend_predictions_mean(bin_pred_d4_cal).Label * blend_predictions_mean(cls_pred_d4_cal).Label,
                ),
            ),
        )
        # print(metric, "Prod  ", prod_pred_d4_cal_score, "*" if prod_pred_d4_cal_score == max_score else "")

        #
        # if bin_pred_d4_score == max_score:
        #     predictions = make_binary_predictions(test_predictions_d4)
        #
        #     predictions = blend_predictions_mean(predictions)
        #     predictions.to_csv(
        #         os.path.join(output_dir, f"mean_{max_score:.6f}_bin_{compute_checksum_v2(fnames_for_checksum)}.csv"),
        #         index=False,
        #     )
        # if bin_pred_d4_cal_score == max_score:
        #     predictions = make_binary_predictions_calibrated(test_predictions_d4, oof_predictions_d4)
        #
        #     predictions = blend_predictions_mean(predictions)
        #     predictions.to_csv(
        #         os.path.join(
        #             output_dir, f"mean_{max_score:.6f}_bin_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
        #         ),
        #         index=False,
        #     )
        # if cls_pred_d4_score == max_score:
        #     predictions = make_classifier_predictions(test_predictions_d4)
        #
        #     predictions = blend_predictions_mean(predictions)
        #     predictions.to_csv(
        #         os.path.join(output_dir, f"mean_{max_score:.6f}_cls_{compute_checksum_v2(fnames_for_checksum)}.csv"),
        #         index=False,
        #     )
        # if cls_pred_d4_cal_score == max_score:
        #     predictions = make_classifier_predictions_calibrated(test_predictions_d4, oof_predictions_d4)
        #
        #     predictions = blend_predictions_mean(predictions)
        #     predictions.to_csv(
        #         os.path.join(
        #             output_dir, f"mean_{max_score:.6f}_cls_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
        #         ),
        #         index=False,
        #     )
        # if prod_pred_d4_cal_score == max_score:
        #     cls_predictions = make_classifier_predictions_calibrated(test_predictions_d4, oof_predictions_d4)
        #     bin_predictions = make_binary_predictions_calibrated(test_predictions_d4, oof_predictions_d4)
        #
        #     predictions1 = blend_predictions_mean(cls_predictions)
        #     predictions2 = blend_predictions_mean(bin_predictions)
        #     predictions = predictions1.copy()
        #     predictions.Label = predictions1.Label * predictions2.Label
        #
        #     predictions.to_csv(
        #         os.path.join(
        #             output_dir, f"mean_{max_score:.6f}_prod_cal_{compute_checksum_v2(fnames_for_checksum)}.csv"
        #         ),
        #         index=False,
        #     )


if __name__ == "__main__":
    main()
