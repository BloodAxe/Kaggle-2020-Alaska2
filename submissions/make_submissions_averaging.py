import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings

from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import (
    make_classifier_predictions,
    make_binary_predictions,
    blend_predictions_mean,
)
from submissions.eval_tta import get_predictions_csv


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

        classifier = make_classifier_predictions(predictions)
        classifier_hv = make_classifier_predictions(predictions_hv)
        classifier_d4 = make_classifier_predictions(predictions_d4)
        blend_classifier_mean = blend_predictions_mean(classifier)
        blend_classifier_mean_hv = blend_predictions_mean(classifier_hv)
        blend_classifier_mean_d4 = blend_predictions_mean(classifier_d4)
        print(metric, "Mean", "Classifier", "  ", alaska_weighted_auc(y_true, blend_classifier_mean.Label))
        print(metric, "Mean", "Classifier", "hv", alaska_weighted_auc(y_true, blend_classifier_mean_hv.Label))
        print(metric, "Mean", "Classifier", "d4", alaska_weighted_auc(y_true, blend_classifier_mean_d4.Label))


if __name__ == "__main__":
    main()
