import os
from collections import defaultdict

import pandas as pd

from alaska2 import alaska_weighted_auc
from alaska2.submissions import calibrated, as_hv_tta, as_d4_tta, classifier_probas, sigmoid, infer_fold
from submissions import ela_skresnext50_32x4d
from submissions import rgb_tf_efficientnet_b2_ns
from submissions import rgb_tf_efficientnet_b6_ns


def get_predictions_csv(experiment, metric: str, type: str, tta: str = None):
    if isinstance(experiment, list):
        return [get_predictions_csv(x, metric, type, tta) for x in experiment]

    assert type in {"test", "holdout"}
    assert metric in {"loss", "bauc", "cauc"}
    assert tta in {None, "d4", "hv"}
    checkpoints_dir = {"loss": "checkpoints", "bauc": "checkpoints_auc", "cauc": "checkpoints_auc_classifier"}[metric]
    csv = os.path.join("models", experiment, "main", checkpoints_dir, f"best_{type}_predictions.csv")
    if tta == "d4":
        csv = as_d4_tta([csv])[0]
    elif tta == "hv":
        csv = as_hv_tta([csv])[0]
    return csv


def main():
    output_dir = os.path.dirname(__file__)

    summary_df = defaultdict(list)

    experiments = [
        "May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        "May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        "May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        "May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
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
        #
        "Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16"
    ]

    all_predictions = [
        (
            "loss",
            get_predictions_csv(experiments, "loss", "test"),
            get_predictions_csv(experiments, "loss", "holdout"),
        ),
        (
            "bauc",
            get_predictions_csv(experiments, "bauc", "test"),
            get_predictions_csv(experiments, "bauc", "holdout"),
        ),
        (
            "cauc",
            get_predictions_csv(experiments, "cauc", "test"),
            get_predictions_csv(experiments, "cauc", "holdout"),
        ),
    ]

    for checkpoint_metric, test_predictions, oof_predictions in all_predictions:

        for oof_p, oof_p_hv_tta, oof_p_d4_tta in zip(
            oof_predictions, as_hv_tta(oof_predictions), as_d4_tta(oof_predictions)
        ):
            path_components = os.path.normpath(oof_p).split(os.sep)

            # No TTA
            summary_df["checkpoint"].append(path_components[1])
            summary_df["fold"].append(infer_fold(oof_p))
            summary_df["metric"].append(checkpoint_metric)

            try:
                df = pd.read_csv(oof_p)
                summary_df["bauc"].append(
                    alaska_weighted_auc(df["true_modification_flag"], df["pred_modification_flag"].apply(sigmoid))
                )
                summary_df["cauc"].append(
                    alaska_weighted_auc(
                        df["true_modification_flag"], df["pred_modification_type"].apply(classifier_probas)
                    )
                )

            except Exception as e:
                print(e)
                for k in ["bauc", "cauc"]:
                    summary_df[k].append("N/A")

            try:
                df = pd.read_csv(oof_p_hv_tta)
                summary_df["bauc (HV tta)"].append(
                    alaska_weighted_auc(df["true_modification_flag"], df["pred_modification_flag"].apply(sigmoid))
                )
                summary_df["cauc (HV tta)"].append(
                    alaska_weighted_auc(
                        df["true_modification_flag"], df["pred_modification_type"].apply(classifier_probas)
                    )
                )

            except Exception as e:
                print(e)
                for k in ["bauc (HV tta)", "cauc (HV tta)"]:
                    summary_df[k].append("N/A")

            try:
                df = pd.read_csv(oof_p_d4_tta)
                summary_df["bauc (D4 tta)"].append(
                    alaska_weighted_auc(df["true_modification_flag"], df["pred_modification_flag"].apply(sigmoid))
                )
                summary_df["cauc (D4 tta)"].append(
                    alaska_weighted_auc(
                        df["true_modification_flag"], df["pred_modification_type"].apply(classifier_probas)
                    )
                )

            except Exception as e:
                print(e)
                for k in ["bauc (D4 tta)", "cauc (D4 tta)"]:
                    summary_df[k].append("N/A")

    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
