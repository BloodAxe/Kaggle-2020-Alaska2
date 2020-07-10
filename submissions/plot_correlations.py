import os
import numpy as np
from pytorch_toolbelt.utils import fs
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

submissions = [x for x in fs.find_in_dir(".") if str.endswith(x, ".csv")]
names = list(map(lambda x: fs.id_from_fname(x)[:32], submissions))
submissions = [pd.read_csv(x).sort_values(by="Id").reset_index() for x in submissions]

cm = np.zeros((len(submissions), len(submissions)))
for i in range(len(submissions)):
    for j in range(len(submissions)):
        cm[i, j] = spearmanr(submissions[i].Label, submissions[j].Label).correlation

print(cm)

plt.figure(figsize=(10 + len(submissions), 10 + len(submissions)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
disp.plot(include_values=True, cmap="Blues", ax=plt.gca(), xticks_rotation="45")
plt.tight_layout()
plt.show()
