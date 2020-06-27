import pandas as pd

from alaska2.submissions import classifier_probas
from alaska2.dataset import OUTPUT_PRED_MODIFICATION_TYPE

dfs = []
for fold_index, fname in enumerate(
    [
        "models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
        "models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
        "models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best_oof_predictions.csv",
    ]
):
    df = pd.read_csv(fname)
    df["fold"] = fold_index
    dfs.append(df)

df = pd.concat(dfs)
df["y_pred"] = df[OUTPUT_PRED_MODIFICATION_TYPE].apply(classifier_probas)
df["error"] = (df["true_modification_flag"] - df["y_pred"]).abs()
df = df.sort_values(by="error", ascending=False)
print(len(df))
print(df.head())

import matplotlib.pyplot as plt

df.to_csv("errors.csv", index=False)

df = df[df["error"] > 1e-2]
plt.figure()
plt.hist(df.loc[df["true_modification_flag"] == 0, "error"], label="negatives")
plt.hist(df.loc[df["true_modification_flag"] == 1, "error"], label="posities")
plt.legend()
plt.show()
