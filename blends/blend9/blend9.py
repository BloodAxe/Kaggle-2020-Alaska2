import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alaska2.submissions import blend_predictions_ranked

v25_xl_NR_moreTTA_b4mish_b2mish_xlmish = (
    pd.read_csv("submission_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish.csv").sort_values(by="Id").reset_index()
)

xgb_cls_gs_09421 = pd.read_csv("xgb_cls_0.9421_Gf0_Gf1_Gf2_Gf3_Hnrmishf2_Hnrmishf1_Kmishf0.csv")
xgb_cls_gs_09421_image_f = pd.read_csv(
    "xgb_cls_gs_0.9421_Gf0_Gf1_Gf2_Gf3_Hnrmishf2_Hnrmishf1_Kmishf0_image_features.csv"
)
embeddings_09413 = pd.read_csv("2nd_level_stacking_0.9413_embeddings_test_submission_d4.csv")
# print(embeddings_09413.describe())

# Force 1.01 value of OOR values in my submission
oor_mask = v25_xl_NR_moreTTA_b4mish_b2mish_xlmish.Label > 1.0

xgb_cls_gs_09421.loc[oor_mask, "Label"] = 1.01
embeddings_09413.loc[oor_mask, "Label"] = 1.01
xgb_cls_gs_09421_image_f.loc[oor_mask, "Label"] = 1.01


submissions = [v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, xgb_cls_gs_09421, xgb_cls_gs_09421_image_f, embeddings_09413]
for s in submissions:
    s["Label"] = s["Label"].astype(np.float32)

cm = np.zeros((len(submissions), len(submissions)))
for i in range(len(submissions)):
    for j in range(len(submissions)):
        cm[i, j] = spearmanr(submissions[i].Label, submissions[j].Label).correlation

print(cm)


#
# blend_9_ranked = blend_predictions_ranked([v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, xgb_cls_gs_09421])
# blend_9_ranked.to_csv(
#     "blend_9_ranked_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish_with_xgb_cls_0.9421_Gf0_Gf1_Gf2_Gf3_Hnrmishf2_Hnrmishf1_Kmishf0.csv",
#     index=False,
# )
#
# blend_9_ranked = blend_predictions_ranked([v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, xgb_cls_gs_09421_image_f])
# blend_9_ranked.to_csv(
#     "blend_9_ranked_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish_with_xgb_cls_0.9421_Gf0_Gf1_Gf2_Gf3_Hnrmishf2_Hnrmishf1_Kmishf0_image_features.csv",
#     index=False,
# )
