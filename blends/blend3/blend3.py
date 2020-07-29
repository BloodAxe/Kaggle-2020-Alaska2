import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
from alaska2.submissions import blend_predictions_ranked, blend_predictions_mean

submission_v25_xl_NR_moreTTA = pd.read_csv("submission_v25_xl_NR_moreTTA.csv").sort_values(by="Id").reset_index()
submission_b6_mean_calibrated = pd.read_csv(
    "662cfbbddf616db0df6f59ee2a96cc20_best_cauc_blend_cls_mean_calibrated_0.9422.csv"
)

# Force 1.01 value of OOR values in my submission
oor_mask = submission_v25_xl_NR_moreTTA.Label > 1.0
submission_b6_mean_calibrated.loc[oor_mask, "Label"] = 1.01
print(spearmanr(submission_v25_xl_NR_moreTTA.Label, submission_b6_mean_calibrated.Label))


blend_3_ranked = blend_predictions_ranked([submission_v25_xl_NR_moreTTA, submission_b6_mean_calibrated])
blend_3_ranked.to_csv("blend_3_ranked_from_v25_xl_NR_moreTTA_and_b6_cauc_mean_calibrated.csv", index=False)

blend_3_mean = blend_predictions_mean([submission_v25_xl_NR_moreTTA, submission_b6_mean_calibrated])
blend_3_mean.to_csv("blend_3_mean_from_v25_xl_NR_moreTTA_and_b6_cauc_mean_calibrated.csv", index=False)
