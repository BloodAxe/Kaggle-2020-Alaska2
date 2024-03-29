import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, ConfusionMatrixDisplay
from alaska2.submissions import blend_predictions_ranked, blend_predictions_mean
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

v25_xl_NR_moreTTA_b4mish_b2mish_xlmish = (
    pd.read_csv("submission_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish.csv").sort_values(by="Id").reset_index()
)

xgb_cls_gs_09420 = pd.read_csv("xgb_cls_0.9420_Gf0_Gf1_Gf2_Gf3_Hnrmishf2_Hnrmishf1_.csv")

# Force 1.01 value of OOR values in my submission
oor_mask = v25_xl_NR_moreTTA_b4mish_b2mish_xlmish.Label > 1.0

xgb_cls_gs_09420.loc[oor_mask, "Label"] = 1.01

submissions = [v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, xgb_cls_gs_09420]

cm = np.zeros((len(submissions), len(submissions)))
for i in range(len(submissions)):
    for j in range(len(submissions)):
        cm[i, j] = spearmanr(submissions[i].Label, submissions[j].Label).correlation

print(cm)


blend_8_ranked = blend_predictions_ranked([v25_xl_NR_moreTTA_b4mish_b2mish_xlmish, xgb_cls_gs_09420])
blend_8_ranked.to_csv(
    "blend_8_ranked_v25_xl_NR_moreTTA_b4mish_b2mish_xlmish_with_xgb_cls_0.9420_Gf0_Gf1_Gf2_Gf3_Hnrmishf2_Hnrmishf1.csv",
    index=False,
)
