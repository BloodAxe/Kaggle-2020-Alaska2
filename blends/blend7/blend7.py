import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, ConfusionMatrixDisplay
from alaska2.submissions import blend_predictions_ranked, blend_predictions_mean
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

v25_xl_NR_moreTTA_b4mish = pd.read_csv("submission_v25_xl_NR_moreTTA_b4mish.csv").sort_values(by="Id").reset_index()

mean_09406 = pd.read_csv("mean_0.9406_cls_Cf2cauc_Df1cauc_Df2cauc_Ff0cauc_Gf0cauc_Gf1cauc_Gf2cauc_Gf3cauc.csv")
xgb_cls_gs_09445 = pd.read_csv(
    "xgb_cls_gs_0.9445_Cf2cauc_Df1cauc_Df2cauc_Ff0cauc_Gf0cauc_Gf1cauc_Gf2cauc_Gf3cauc_.csv"
)

# Force 1.01 value of OOR values in my submission
oor_mask = v25_xl_NR_moreTTA_b4mish.Label > 1.0

mean_09406.loc[oor_mask, "Label"] = 1.01
xgb_cls_gs_09445.loc[oor_mask, "Label"] = 1.01

submissions = [v25_xl_NR_moreTTA_b4mish, v25_xl_NR_moreTTA_b4mish, mean_09406, xgb_cls_gs_09445]

cm = np.zeros((len(submissions), len(submissions)))
for i in range(len(submissions)):
    for j in range(len(submissions)):
        cm[i, j] = spearmanr(submissions[i].Label, submissions[j].Label).correlation

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["v25_xl_NR_moreTTA", "v25_xl_NR_moreTTA_b4mish", "mean_09406", "xgb_cls_gs_09445"],
)
plt.figure(figsize=(8, 8))
disp.plot(include_values=True, cmap="Blues", ax=plt.gca(), xticks_rotation=45)
plt.show()

blend_6_ranked = blend_predictions_ranked([v25_xl_NR_moreTTA_b4mish, xgb_cls_gs_09445])
blend_6_ranked.to_csv("blend_7_ranked_v25_xl_NR_moreTTA_b4mish_with_xgb_cls_gs_09445.csv", index=False)
