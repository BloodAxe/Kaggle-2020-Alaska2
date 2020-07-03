import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
from alaska2.submissions import blend_predictions_ranked, blend_predictions_mean

submission_v25_xl_NR_moreTTA = pd.read_csv("submission_v25_xl_NR_moreTTA.csv").sort_values(by="Id")
stacked_b6_xgb_cv = pd.read_csv("662cfbbddf616db0df6f59ee2a96cc20_xgb_cv_0.9485.csv")


print(spearmanr(submission_v25_xl_NR_moreTTA.Label, stacked_b6_xgb_cv.Label))


blend_1_ranked = blend_predictions_ranked([submission_v25_xl_NR_moreTTA, stacked_b6_xgb_cv])
blend_1_ranked.to_csv("blend_1_ranked.csv", index=False)

blend_1_mean = blend_predictions_mean([submission_v25_xl_NR_moreTTA, stacked_b6_xgb_cv])
blend_1_mean.to_csv("blend_1_mean.csv", index=False)
