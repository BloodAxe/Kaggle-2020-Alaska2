import os
from collections import defaultdict

import pandas as pd

from alaska2.submissions import calibrated, as_hv_tta, as_d4_tta
from submissions import ela_skresnext50_32x4d
from submissions import rgb_tf_efficientnet_b2_ns
from submissions import rgb_tf_efficientnet_b6_ns


def main():
    output_dir = os.path.dirname(__file__)

    summary_df = defaultdict(list)

    all_predictions = [
        # (
        #     "loss",
        #     ela_skresnext50_32x4d.ela_skresnext50_32x4d_best_loss,
        #     ela_skresnext50_32x4d.ela_skresnext50_32x4d_best_loss_oof,
        # ),
        # (
        #     "b_auc",
        #     ela_skresnext50_32x4d.ela_skresnext50_32x4d_best_auc_b,
        #     ela_skresnext50_32x4d.ela_skresnext50_32x4d_best_auc_c_oof,
        # ),
        # (
        #     "c_auc",
        #     ela_skresnext50_32x4d.ela_skresnext50_32x4d_best_auc_c,
        #     ela_skresnext50_32x4d.ela_skresnext50_32x4d_best_auc_c_oof,
        # ),
        # # B6
        # (
        #     "loss",
        #     rgb_tf_efficientnet_b6_ns.rgb_tf_efficientnet_b6_ns_best_loss,
        #     rgb_tf_efficientnet_b6_ns.rgb_tf_efficientnet_b6_ns_best_loss_oof,
        # ),
        # (
        #     "b_auc",
        #     rgb_tf_efficientnet_b6_ns.rgb_tf_efficientnet_b6_ns_best_auc_b,
        #     rgb_tf_efficientnet_b6_ns.rgb_tf_efficientnet_b6_ns_best_auc_b_oof,
        # ),
        # (
        #     "c_auc",
        #     rgb_tf_efficientnet_b6_ns.rgb_tf_efficientnet_b6_ns_best_auc_c,
        #     rgb_tf_efficientnet_b6_ns.rgb_tf_efficientnet_b6_ns_best_auc_c_oof,
        # ),
        # # B2
        # (
        #     "c_auc",
        #     rgb_tf_efficientnet_b2_ns.rgb_tf_efficientnet_b2_ns_best_auc_c,
        #     rgb_tf_efficientnet_b2_ns.rgb_tf_efficientnet_b2_ns_best_auc_c_oof,
        # ),
        # New
        (
            "loss",
            [
                # "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
                "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_test_predictions.csv",
            ],
            [
                # "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
                "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best_oof_predictions.csv",
            ],
        ),
        (
            "c_auc",
            [
                "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
                "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_test_predictions.csv",
            ],
            [
                "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
                "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
            ],
        ),
    ]

    for checkpoint_metric, test_predictions, oof_predictions in all_predictions:
        keys = ["b_auc_before", "b_auc_after", "c_auc_before", "c_auc_after"]

        for test_p, oof_p in zip(test_predictions, oof_predictions):
            # No TTA
            summary_df["test_predictions"].append(test_p)
            summary_df["checkpoint_metric"].append(checkpoint_metric)
            summary_df["tta"].append("none")
            try:
                _, score = calibrated(pd.read_csv(test_p), pd.read_csv(oof_p))
                for k in keys:
                    summary_df[k].append(score[k])
            except Exception as e:
                print(e)
                for k in keys:
                    summary_df[k].append("N/A")

        # HV TTA
        for test_p, oof_p in zip(as_hv_tta(test_predictions), as_hv_tta(oof_predictions)):
            summary_df["test_predictions"].append(test_p)
            summary_df["checkpoint_metric"].append(checkpoint_metric)
            summary_df["tta"].append("hv")
            try:
                _, score = calibrated(pd.read_csv(test_p), pd.read_csv(oof_p))
                for k in keys:
                    summary_df[k].append(score[k])
            except Exception as e:
                print(e)
                for k in keys:
                    summary_df[k].append("N/A")

        # D4 TTA
        for test_p, oof_p in zip(as_d4_tta(test_predictions), as_d4_tta(oof_predictions)):
            summary_df["test_predictions"].append(test_p)
            summary_df["checkpoint_metric"].append(checkpoint_metric)
            summary_df["tta"].append("d4")
            try:
                _, score = calibrated(pd.read_csv(test_p), pd.read_csv(oof_p))
                for k in keys:
                    summary_df[k].append(score[k])
            except Exception as e:
                print(e)
                for k in keys:
                    summary_df[k].append("N/A")

    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
