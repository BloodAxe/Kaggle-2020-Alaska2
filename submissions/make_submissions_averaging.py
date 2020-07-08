import json
import os
import re
from typing import List

import numpy as np
from pytorch_toolbelt.utils import fs
from sklearn.isotonic import IsotonicRegression as IR

from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import (
    make_classifier_predictions,
    make_classifier_predictions_calibrated,
    make_binary_predictions,
    make_binary_predictions_calibrated,
    blend_predictions_mean,
)
from submissions.eval_tta import get_predictions_csv


# Used to ignore warnings generated from StackingCVClassifier


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
            .replace("fold", "f")
            .replace("local_rank_0", "")
            .replace("tf_efficientnet_b2_ns", "B2")
            .replace("tf_efficientnet_b3_ns", "B3")
            .replace("tf_efficientnet_b6_ns", "B6")
            .replace("tf_efficientnet_b7_ns", "B7")
        )
        x = re.sub(r"\w{3}\d{2}_\d{2}_\d{2}", "", x).replace("_", "")
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
        "C_Jun24_22_00_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        "D_Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        # "D_Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        # "E_Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "E_Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
        #
        "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
    ]

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
        y_true = bin_pred_d4[0].y_true.values

        bin_pred_d4_score = alaska_weighted_auc(y_true, blend_predictions_mean(bin_pred_d4).Label)

        cls_pred_d4 = make_classifier_predictions(predictions_d4)
        cls_pred_d4_score = alaska_weighted_auc(y_true, blend_predictions_mean(cls_pred_d4).Label)

        bin_pred_d4_cal = make_binary_predictions_calibrated(predictions_d4, oof_predictions_d4)
        bin_pred_d4_cal_score = alaska_weighted_auc(y_true, blend_predictions_mean(bin_pred_d4_cal).Label)

        cls_pred_d4_cal = make_classifier_predictions_calibrated(predictions_d4, oof_predictions_d4)
        cls_pred_d4_cal_score = alaska_weighted_auc(y_true, blend_predictions_mean(cls_pred_d4_cal).Label)

        print(metric, "Bin NC", "d4", bin_pred_d4_score)
        print(metric, "Bin CL", "d4", cls_pred_d4_score)
        print(metric, "Cls NC", "d4", bin_pred_d4_cal_score)
        print(metric, "Cls CL", "d4", cls_pred_d4_cal_score)

        blend_cls_d4 = blend_predictions_mean(cls_pred_d4)

        ir_type = IR(out_of_bounds="clip", y_min=0, y_max=1)
        y_pred_raw = blend_cls_d4.Label.values
        c_auc_before = alaska_weighted_auc(y_true, y_pred_raw)
        y_pred_cal = ir_type.fit_transform(y_pred_raw, y_true)
        c_auc_after = alaska_weighted_auc(y_true, y_pred_cal)

        print(metric, "Calibrated after blend", c_auc_before, c_auc_after)

        max_score = max(bin_pred_d4_score, cls_pred_d4_score, bin_pred_d4_cal_score, cls_pred_d4_cal_score)
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


if __name__ == "__main__":
    main()
