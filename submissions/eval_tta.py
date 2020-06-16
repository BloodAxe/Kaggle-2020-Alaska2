import os
from collections import defaultdict

import pandas as pd

from alaska2 import alaska_weighted_auc
from alaska2.submissions import calibrated, as_hv_tta, as_d4_tta, classifier_probas
from submissions import ela_skresnext50_32x4d
from submissions import rgb_tf_efficientnet_b2_ns
from submissions import rgb_tf_efficientnet_b6_ns


def main():
    output_dir = os.path.dirname(__file__)

    summary_df = defaultdict(list)

    best_loss = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
        "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
        "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
    ]
    best_bauc = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
        "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
        "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best_test_predictions.csv",
    ]
    best_cauc = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
        "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
        "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
    ]

    best_loss_h = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
        "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
        "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best_holdout_predictions.csv",
    ]
    best_bauc_h = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
        "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
        "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best_holdout_predictions.csv",
    ]
    best_cauc_h = [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
        "models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
        "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_holdout_predictions.csv",
    ]

    all_predictions = [
        ("loss", best_loss, best_loss_h),
        ("b_auc", best_bauc, best_bauc_h),
        ("c_auc", best_cauc, best_cauc_h),
    ]

    for checkpoint_metric, test_predictions, oof_predictions in all_predictions:
        keys = ["b_auc_score", "c_auc_score"]

        for test_p, oof_p in zip(test_predictions, oof_predictions):
            # No TTA
            summary_df["test_predictions"].append(test_p.split('/')[1])
            summary_df["checkpoint_metric"].append(checkpoint_metric)
            summary_df["tta"].append("none")
            try:
                df = pd.read_csv(oof_p)
                summary_df["b_auc_score"].append(
                    alaska_weighted_auc(df["true_modification_flag"], df["pred_modification_flag"])
                )
                summary_df["c_auc_score"].append(
                    alaska_weighted_auc(
                        df["true_modification_flag"], df["pred_modification_type"].apply(classifier_probas)
                    )
                )

            except Exception as e:
                print(e)
                for k in keys:
                    summary_df[k].append("N/A")

        # HV TTA
        for test_p, oof_p in zip(as_hv_tta(test_predictions), as_hv_tta(oof_predictions)):
            summary_df["test_predictions"].append(test_p.split('/')[1])
            summary_df["checkpoint_metric"].append(checkpoint_metric)
            summary_df["tta"].append("hv")
            try:
                df = pd.read_csv(oof_p)
                summary_df["b_auc_score"].append(
                    alaska_weighted_auc(df["true_modification_flag"], df["pred_modification_flag"])
                )
                summary_df["c_auc_score"].append(
                    alaska_weighted_auc(
                        df["true_modification_flag"], df["pred_modification_type"].apply(classifier_probas)
                    )
                )

            except Exception as e:
                print(e)
                for k in keys:
                    summary_df[k].append("N/A")

        # D4 TTA
        for test_p, oof_p in zip(as_d4_tta(test_predictions), as_d4_tta(oof_predictions)):
            summary_df["test_predictions"].append(test_p.split('/')[1])
            summary_df["checkpoint_metric"].append(checkpoint_metric)
            summary_df["tta"].append("d4")
            try:
                df = pd.read_csv(oof_p)
                summary_df["b_auc_score"].append(
                    alaska_weighted_auc(df["true_modification_flag"], df["pred_modification_flag"])
                )
                summary_df["c_auc_score"].append(
                    alaska_weighted_auc(
                        df["true_modification_flag"], df["pred_modification_type"].apply(classifier_probas)
                    )
                )

            except Exception as e:
                print(e)
                for k in keys:
                    summary_df[k].append("N/A")

    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
