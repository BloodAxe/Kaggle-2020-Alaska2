import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
from alaska2.submissions import blend_predictions_ranked, blend_predictions_mean

submission_v25_xl_NR_moreTTA = pd.read_csv("submission_v25_xl_NR_moreTTA.csv").sort_values(by="Id").reset_index()
submission_b6_mean_calibrated = pd.read_csv(
    "mean_0.9391_cls_cal_BrgbB6f0cauc_BrgbB6f1cauc_BrgbB6f2cauc_BrgbB6f3cauc_CrgbB2f2cauc_DrgbB7f1cauc_DrgbB7f2cauc_ErgbB6f0istego100kcauc_FrgbB3f0cauc.csv"
)
submission_b6_cmb_uncalibrated = pd.read_csv(
    "cmb_0.9414_5_B_rgb_B6_f3_cauc_bin_C_rgb_B2_f2_cauc_bin_C_rgb_B2_f2_cauc_cls_D_rgb_B7_f2_cauc_cls_E_rgb_B6_f0_istego100k_cauc_cls.csv"
)
submission_b6_xgb = pd.read_csv(
    "xgb_gs_0.9434_BrgbB6f0cauc_BrgbB6f1cauc_BrgbB6f2cauc_BrgbB6f3cauc_CrgbB2f2cauc_DrgbB7f1cauc_DrgbB7f2cauc_ErgbB6f0istego100kcauc_FrgbB3f0cauc.csv"
)

# Force 1.01 value of OOR values in my submission
oor_mask = submission_v25_xl_NR_moreTTA.Label > 1.0
submission_b6_mean_calibrated.loc[oor_mask, "Label"] = 1.01
submission_b6_cmb_uncalibrated.loc[oor_mask, "Label"] = 1.01
submission_b6_xgb.loc[oor_mask, "Label"] = 1.01

print(spearmanr(submission_v25_xl_NR_moreTTA.Label, submission_b6_cmb_uncalibrated.Label))
print(spearmanr(submission_v25_xl_NR_moreTTA.Label, submission_b6_mean_calibrated.Label))
print(spearmanr(submission_v25_xl_NR_moreTTA.Label, submission_b6_xgb.Label))

#
# blend_4_ranked = blend_predictions_ranked([submission_v25_xl_NR_moreTTA, submission_b6_mean_calibrated])
# blend_4_ranked.to_csv("blend_3_ranked_from_v25_xl_NR_moreTTA_and_mean_0.9391_cls_cal_BrgbB6f0cauc_BrgbB6f1cauc_BrgbB6f2cauc_BrgbB6f3cauc_CrgbB2f2cauc_DrgbB7f1cauc_DrgbB7f2cauc_ErgbB6f0istego100kcauc_FrgbB3f0cauc.csv", index=False)

# blend_4_ranked = blend_predictions_ranked([submission_v25_xl_NR_moreTTA, submission_b6_mean_calibrated])
# blend_4_ranked.to_csv("blend_3_ranked_from_v25_xl_NR_moreTTA_and_mean_0.9391_cls_cal_BrgbB6f0cauc_BrgbB6f1cauc_BrgbB6f2cauc_BrgbB6f3cauc_CrgbB2f2cauc_DrgbB7f1cauc_DrgbB7f2cauc_ErgbB6f0istego100kcauc_FrgbB3f0cauc.csv", index=False)

blend_4_mean = blend_predictions_ranked([submission_v25_xl_NR_moreTTA, submission_b6_xgb])
blend_4_mean.to_csv(
    "blend_4_ranked_from_v25_xl_NR_moreTTA_and_xgb_gs_0.9434_BrgbB6f0cauc_BrgbB6f1cauc_BrgbB6f2cauc_BrgbB6f3cauc_CrgbB2f2cauc_DrgbB7f1cauc_DrgbB7f2cauc_ErgbB6f0istego100kcauc_FrgbB3f0cauc.csv",
    index=False,
)
