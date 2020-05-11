import pandas as pd
import numpy as np
from pytorch_toolbelt.utils import to_numpy
import matplotlib.pyplot as plt
from alaska2 import *

df = pd.read_csv("models\May08_22_42_rgb_resnet34_fold1\main\checkpoints_auc\\train.100_oof_predictions.csv")
print(df.head())

y_true = df[INPUT_TRUE_MODIFICATION_FLAG].values.astype(int)
y_pred_logits = df[OUTPUT_PRED_MODIFICATION_FLAG].values

scores = []
temperatures = np.arange(1, 1.8, 0.01)
offsets = np.arange(-5, 5, 0.1)
# for t in temperatures:
#     o = 0
#     y_pred = torch.from_numpy(y_pred_logits).add(o).mul(t).sigmoid()
#     s = alaska_weighted_auc(y_true, to_numpy(y_pred))
#     scores.append(s)

for o in offsets:
    o = 0
    t = 1
    y_pred = torch.from_numpy(y_pred_logits).add(o).mul(t).sigmoid()
    s = alaska_weighted_auc(y_true, to_numpy(y_pred))
    scores.append(s)


plt.figure()
plt.plot(offsets, scores)
plt.show()
