import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ConfusionMatrixDisplay

from alaska2.submissions import blend_predictions_ranked

v25_xl_NR_moreTTA_b4mish_b2mish_xlmish = (
    pd.read_csv("submission_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish.csv").sort_values(by="Id").reset_index()
)

v26_dctr_jrm_srnet_mns_mnxlm_b2_b4m_b5m_catboost_d5 = (
    pd.read_csv("submission_v26_dctr_jrm_srnet_mns_mnxlm_b2_b4m_b5m_catboost_d5_VAL_saved.csv")
    .sort_values(by="Id")
    .reset_index()
)

embeddings_09411 = pd.read_csv("2nd_level_stacking_0.9411_embeddings_test_submission_d4.csv")
mean_0_9417 = pd.read_csv("mean_0.9417_cls_Kmishf0cauc_Jnrmishf1cauc_Hnrmishf2cauc_Kmishf3cauc.csv")
cmb_0_9424 = pd.read_csv(
    "cmb_mean_0.9424_7_Gf3caucbin_Hnrmishf1caucbin_Kmishf3caucbin_Hnrmishf1cauccls_Kmishf0cauccls_Kmishf3cauccls_Jnrmishf1cauccls.csv"
)

xgb_0_9424 = pd.read_csv("xgb_cls_gs_0.9424_Gf0_Gf1_Gf2_Gf3_Kmishf0_Jnrmishf1_Hnrmishf2_Kmishf3_with_logits.csv")
lgb_0_9421 = pd.read_csv("lgbm_gs_0.9421_Gf0_Gf1_Gf2_Gf3_Kmishf0_Jnrmishf1_Hnrmishf2_Kmishf3.csv")

# Force 1.01 value of OOR values in my submission
oor_mask = v25_xl_NR_moreTTA_b4mish_b2mish_xlmish.Label > 1.0

embeddings_09411.loc[oor_mask, "Label"] = 1.01
mean_0_9417.loc[oor_mask, "Label"] = 1.01
cmb_0_9424.loc[oor_mask, "Label"] = 1.01
xgb_0_9424.loc[oor_mask, "Label"] = 1.01
lgb_0_9421.loc[oor_mask, "Label"] = 1.01

submissions = [
    v25_xl_NR_moreTTA_b4mish_b2mish_xlmish,
    v26_dctr_jrm_srnet_mns_mnxlm_b2_b4m_b5m_catboost_d5,
    embeddings_09411,
    mean_0_9417,
    cmb_0_9424,
    xgb_0_9424,
    lgb_0_9421,
]
for s in submissions:
    s["Label"] = s["Label"].astype(np.float32)

cm = np.zeros((len(submissions), len(submissions)))
for i in range(len(submissions)):
    for j in range(len(submissions)):
        cm[i, j] = spearmanr(submissions[i].Label, submissions[j].Label).correlation

print(cm)

import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["v25", "v26", "emb_09411", "avg_0_9417", "cmb_0_9424", "xgb_0_9424", "lgb_0_9421"],
)
plt.figure(figsize=(8, 8))
disp.plot(include_values=True, cmap="Blues", ax=plt.gca(), xticks_rotation=45)
plt.savefig(fname="predictions_corr.png")
plt.show()


# Submit 1 - v25 + embedding
# Submit 2 - v25 + tuned models
# Submit 3 - v26 + tuned models
# Submit 4 -
# Submit 5 -

blend_10_ranked = blend_predictions_ranked([v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, embeddings_09411])
print(blend_10_ranked.describe())
blend_10_ranked.to_csv("blend_10_ranked_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish_with_embeddings_09411.csv", index=False)

blend_10_ranked = blend_predictions_ranked([v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, mean_0_9417])
print(blend_10_ranked.describe())
blend_10_ranked.to_csv(
    "blend_10_ranked_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish_with_mean_0.9417_cls_Kmishf0cauc_Jnrmishf1cauc_Hnrmishf2cauc_Kmishf3cauc.csv",
    index=False,
)

blend_10_ranked = blend_predictions_ranked([v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, xgb_0_9424])
print(blend_10_ranked.describe())
blend_10_ranked.to_csv(
    "blend_10_ranked_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish_with_xgb_cls_gs_0.9424_Gf0_Gf1_Gf2_Gf3_Kmishf0_Jnrmishf1_Hnrmishf2_Kmishf3_with_logits.csv",
    index=False,
)


blend_10_ranked = blend_predictions_ranked([v26_dctr_jrm_srnet_mns_mnxlm_b2_b4m_b5m_catboost_d5, xgb_0_9424])
print(blend_10_ranked.describe())
blend_10_ranked.to_csv(
    "blend_10_ranked_v26_dctr_jrm_srnet_mns_mnxlm_b2_b4m_b5m_catboost_d5_with_xgb_cls_gs_0.9424_Gf0_Gf1_Gf2_Gf3_Kmishf0_Jnrmishf1_Hnrmishf2_Kmishf3_with_logits.csv",
    index=False,
)


blend_10_ranked = blend_predictions_ranked(
    [
        v25_xl_NR_moreTTA_b4mish_b2mish_xlmish,
        v26_dctr_jrm_srnet_mns_mnxlm_b2_b4m_b5m_catboost_d5,
        xgb_0_9424,
        lgb_0_9421,
    ]
)
print(blend_10_ranked.describe())
blend_10_ranked.to_csv("blend_10_ranked_v25_v26_with_xgb_lgmb.csv", index=False)
