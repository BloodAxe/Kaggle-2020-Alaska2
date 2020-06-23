import json
import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings
from hashlib import md5

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


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        #
        "Jun02_12_26_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        "Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        "Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        "Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        "Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
    ]

    if True:
        for metric in ["loss", "bauc", "cauc"]:
            predictions = get_predictions_csv(experiments, metric, "holdout")
            predictions_hv = get_predictions_csv(experiments, metric, "holdout", "hv")
            predictions_d4 = get_predictions_csv(experiments, metric, "holdout", "d4")

            binary_predictions = make_binary_predictions(predictions)
            binary_predictions_hv = make_binary_predictions(predictions_hv)
            binary_predictions_d4 = make_binary_predictions(predictions_d4)
            y_true = binary_predictions[0].y_true.values

            blend_binary_mean = blend_predictions_mean(binary_predictions)
            blend_binary_mean_hv = blend_predictions_mean(binary_predictions_hv)
            blend_binary_mean_d4 = blend_predictions_mean(binary_predictions_d4)
            print(metric, "Mean", "Binary", "  ", alaska_weighted_auc(y_true, blend_binary_mean.Label))
            print(metric, "Mean", "Binary", "hv", alaska_weighted_auc(y_true, blend_binary_mean_hv.Label))
            print(metric, "Mean", "Binary", "d4", alaska_weighted_auc(y_true, blend_binary_mean_d4.Label))

            cls_pred = make_classifier_predictions(predictions)
            cls_pred_hv = make_classifier_predictions(predictions_hv)
            cls_pred_d4 = make_classifier_predictions(predictions_d4)
            blend_classifier_mean = blend_predictions_mean(cls_pred)
            blend_classifier_mean_hv = blend_predictions_mean(cls_pred_hv)
            blend_classifier_mean_d4 = blend_predictions_mean(cls_pred_d4)
            print(metric, "Mean", "Classifier", "  ", alaska_weighted_auc(y_true, blend_classifier_mean.Label))
            print(metric, "Mean", "Classifier", "hv", alaska_weighted_auc(y_true, blend_classifier_mean_hv.Label))
            print(metric, "Mean", "Classifier", "d4", alaska_weighted_auc(y_true, blend_classifier_mean_d4.Label))

            cls_pred = make_classifier_predictions_calibrated(predictions, predictions)
            cls_pred_hv = make_classifier_predictions_calibrated(predictions_hv, predictions_hv)
            cls_pred_d4 = make_classifier_predictions_calibrated(predictions_d4, predictions_d4)
            blend_classifier_mean = blend_predictions_mean(cls_pred)
            blend_classifier_mean_hv = blend_predictions_mean(cls_pred_hv)
            blend_classifier_mean_d4 = blend_predictions_mean(cls_pred_d4)
            print(metric, "Mean", "Classifier calibrated", "  ", alaska_weighted_auc(y_true, blend_classifier_mean.Label))
            print(metric, "Mean", "Classifier calibrated", "hv", alaska_weighted_auc(y_true, blend_classifier_mean_hv.Label))
            print(metric, "Mean", "Classifier calibrated", "d4", alaska_weighted_auc(y_true, blend_classifier_mean_d4.Label))

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
    test_predictions_d4 = get_predictions_csv(experiments, "loss", "test", "d4")
    checksum = compute_checksum(test_predictions_d4)
    test_predictions_d4 = make_classifier_predictions(test_predictions_d4)
    test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    cv_score = 0.9377
    test_predictions_d4.to_csv(
        os.path.join(output_dir, f"{checksum}_best_loss_blend_cls_mean_{cv_score}.csv"), index=False
    )

    test_predictions_d4 = get_predictions_csv(experiments, "bauc", "test", "d4")
    checksum = compute_checksum(test_predictions_d4)
    test_predictions_d4 = make_binary_predictions(test_predictions_d4)
    test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    cv_score = 0.9386
    test_predictions_d4.to_csv(
        os.path.join(output_dir, f"{checksum}_best_bauc_blend_bin_mean_{cv_score}.csv"), index=False
    )

    test_predictions_d4 = get_predictions_csv(experiments, "cauc", "test", "d4")
    checksum = compute_checksum(test_predictions_d4)
    test_predictions_d4 = make_classifier_predictions(test_predictions_d4)
    test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    cv_score = 0.9388
    test_predictions_d4.to_csv(
        os.path.join(output_dir, f"{checksum}_best_cauc_blend_cls_mean_{cv_score}.csv"), index=False
    )

    test_predictions_d4 = get_predictions_csv(experiments, "cauc", "test", "d4")
    checksum = compute_checksum(test_predictions_d4)
    test_predictions_d4 = make_classifier_predictions_calibrated(test_predictions_d4, get_predictions_csv(experiments, "cauc", "holdout", "d4"))
    test_predictions_d4 = blend_predictions_mean(test_predictions_d4)
    cv_score = 0.9388
    test_predictions_d4.to_csv(
        os.path.join(output_dir, f"{checksum}_best_cauc_blend_cls_mean_calibrated_{cv_score}.csv"), index=False
    )


if __name__ == "__main__":
    main()
