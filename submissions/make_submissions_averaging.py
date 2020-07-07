import json
import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings
from hashlib import md5

from typing import List

from pytorch_toolbelt.utils import fs

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


def compute_checksum(*input):
    object_to_serialize = dict((f"input_{i}", x) for i, x in enumerate(input))
    str_object = json.dumps(object_to_serialize)

    import hashlib

    return hashlib.md5(str_object.encode("utf-8")).hexdigest()


def compute_checksum_v2(fnames: List[str]):
    def sanitize_fname(x):
        x = fs.id_from_fname(x)
        x = (
            x.replace("fp16", "")
            .replace("fold", "F")
            .replace("local_rank_0", "")
            .replace("tf_efficientnet_b2_ns", "B2")
            .replace("tf_efficientnet_b3_ns", "B3")
            .replace("tf_efficientnet_b6_ns", "B6")
            .replace("tf_efficientnet_b7_ns", "B7")
            .replace("__","_")
        )
        return x

    fnames = [sanitize_fname(x) for x in fnames]
    return "_".join(fnames)


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
        # "C_Jun02_12_26_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        # "C_Jun24_22_00_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        # "D_Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        # "D_Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        # "E_Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "E_Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
        #
        # TODO: Compute holdout
        # "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
    ]

    if True:
        for metric in [  # "loss",
            # "bauc",
            "cauc"
        ]:
            predictions_d4 = get_predictions_csv(experiments, metric, "holdout", "d4")
            oof_predictions_d4 = get_predictions_csv(experiments, metric, "oof", "d4")

            bin_pred_d4 = make_binary_predictions(predictions_d4)
            cls_pred_d4 = make_classifier_predictions(predictions_d4)
            y_true = bin_pred_d4[0].y_true.values

            bin_pred_d4_cal = make_binary_predictions_calibrated(predictions_d4, oof_predictions_d4)
            cls_pred_d4_cal = make_classifier_predictions_calibrated(predictions_d4, oof_predictions_d4)

            # print(metric, "Mean", "Bin", "d4", alaska_weighted_auc(y_true, blend_predictions_mean(bin_pred_d4).Label))

            blend_cls_d4 = blend_predictions_mean(cls_pred_d4)

            print(metric, "Mean", "Cls", "d4", alaska_weighted_auc(y_true, blend_cls_d4.Label))

            # from sklearn.isotonic import IsotonicRegression as IR

            # ir_type = IR(out_of_bounds="clip", y_min=0, y_max=1)
            # y_pred_raw = blend_cls_d4.Label.values
            # c_auc_before = alaska_weighted_auc(y_true, y_pred_raw)

            # import numpy as np
            # order = np.argsort(y_pred_raw)

            # y_pred_cal = ir_type.fit_transform(y_pred_raw, y_true)
            # c_auc_after = alaska_weighted_auc(y_true[order], y_pred_cal)

            # print(metric, "Calibrated after blend", c_auc_before, c_auc_after)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.hist(y_pred_raw, alpha=0.5, bins=100, label=f"non-calibrated {c_auc_before}")
            # plt.hist(y_pred_cal, alpha=0.5, bins=100, label=f"calibrated {c_auc_after}")
            # plt.yscale("log")
            # plt.legend()
            # plt.show()

            # print(metric,
            #     "Mean",
            #     "Bin cal.",
            #     "d4",
            #     alaska_weighted_auc(y_true, blend_predictions_mean(bin_pred_d4_cal).Label),
            # )
            print(
                metric,
                "Mean",
                "Cls cal.",
                "d4",
                alaska_weighted_auc(y_true, blend_predictions_mean(cls_pred_d4_cal).Label),
            )

            # cls_pred = make_classifier_predictions(predictions)
            # cls_pred_hv = make_classifier_predictions(predictions_hv)
            # cls_pred_d4 = make_classifier_predictions(predictions_d4)
            # blend_classifier_ranked = blend_predictions_ranked(cls_pred)
            # blend_classifier_ranked_hv = blend_predictions_ranked(cls_pred_hv)
            # blend_classifier_ranked_d4 = blend_predictions_ranked(cls_pred_d4)
            # print(metric, "Ranked", "Classifier", "  ", alaska_weighted_auc(y_true, blend_classifier_ranked.Label))
            # print(metric, "Ranked", "Classifier", "hv", alaska_weighted_auc(y_true, blend_classifier_ranked_hv.Label))
            # print(metric, "Ranked", "Classifier", "d4", alaska_weighted_auc(y_true, blend_classifier_ranked_d4.Label))

            # blend_both_mean = blend_predictions_mean(cls_pred + binary_predictions)
            # blend_both_mean_hv = blend_predictions_mean(cls_pred_hv + binary_predictions_hv)
            # blend_both_mean_d4 = blend_predictions_mean(cls_pred_d4 + binary_predictions_d4)
            # print(metric, "Mean", "Both", "  ", alaska_weighted_auc(y_true, blend_both_mean.Label))
            # print(metric, "Mean", "Both", "hv", alaska_weighted_auc(y_true, blend_both_mean_hv.Label))
            # print(metric, "Mean", "Both", "d4", alaska_weighted_auc(y_true, blend_both_mean_d4.Label))

    # TODO: Make automatic
    # test_predictions_d4 = get_predictions_csv(experiments, "loss", "test", "d4")
    # checksum = compute_checksum(test_predictions_d4)
    # test_predictions_d4 = make_classifier_predictions(test_predictions_d4)
    # test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    # cv_score = 0.9377
    # test_predictions_d4.to_csv(
    #     os.path.join(output_dir, f"{checksum}_best_loss_blend_cls_mean_{cv_score}.csv"), index=False
    # )
    #
    # test_predictions_d4 = get_predictions_csv(experiments, "bauc", "test", "d4")
    # checksum = compute_checksum(test_predictions_d4)
    # test_predictions_d4 = make_binary_predictions(test_predictions_d4)
    # test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    # cv_score = 0.9386
    # test_predictions_d4.to_csv(
    #     os.path.join(output_dir, f"{checksum}_best_bauc_blend_bin_mean_{cv_score}.csv"), index=False
    # )
    #
    # test_predictions_d4 = get_predictions_csv(experiments, "cauc", "test", "d4")
    # checksum = compute_checksum(test_predictions_d4)
    # test_predictions_d4 = make_classifier_predictions(test_predictions_d4)
    # test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    # cv_score = 0.9388
    # test_predictions_d4.to_csv(
    #     os.path.join(output_dir, f"{checksum}_best_cauc_blend_cls_mean_{cv_score}.csv"), index=False
    # )

    # test_predictions_d4 = get_predictions_csv(experiments, "cauc", "test", "d4")
    # checksum = compute_checksum(test_predictions_d4)
    # test_predictions_d4 = make_classifier_predictions_calibrated(
    #     test_predictions_d4, get_predictions_csv(experiments, "cauc", "holdout", "d4")
    # )
    # test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    # cv_score = 0.9429
    # test_predictions_d4.to_csv(
    #     os.path.join(output_dir, f"{checksum}_best_cauc_blend_cls_mean_calibrated_{cv_score}.csv"), index=False
    # )
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.hist(test_predictions_d4.Label, alpha=0.5, bins=100, label="calibrated")
    #
    # test_predictions_d4 = get_predictions_csv(experiments, "cauc", "test", "d4")
    # checksum = compute_checksum(test_predictions_d4)
    # test_predictions_d4 = make_classifier_predictions(test_predictions_d4)
    # test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    # cv_score = 0.9411
    # test_predictions_d4.to_csv(
    #     os.path.join(output_dir, f"{checksum}_best_cauc_blend_cls_mean_{cv_score}.csv"), index=False
    # )
    #
    # plt.hist(test_predictions_d4.Label, alpha=0.5, bins=100, label="non-calibrated")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
