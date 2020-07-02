import pandas as pd
import alaska2

df = pd.read_csv("runs/Jul02_11_33_rgb_tf_efficientnet_b3_ns_fold0/main/checkpoints_auc/last_holdout_predictions.csv")

import matplotlib.pyplot as plt

plt.figure()
plt.hist(df.pred_modification_flag.values)
plt.show()

print(alaska2.alaska_weighted_auc(df.true_modification_flag.values.astype(int), df.pred_modification_flag.values))

print(alaska2.alaska_weighted_auc(df.true_modification_flag.values.astype(int), -df.pred_modification_flag.values))
