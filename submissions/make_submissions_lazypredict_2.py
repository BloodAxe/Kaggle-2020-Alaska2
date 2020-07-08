import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings
from collections import defaultdict

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alaska2 import INPUT_IMAGE_KEY, get_test_dataset

# Classifiers

warnings.simplefilter("ignore")


def main():
    output_dir = os.path.dirname(__file__)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    image_id = [os.path.basename(x) for x in test_ds.images]

    models = [pd.read_csv(os.path.join(output_dir, f"lazypredict_models_{x}.csv")) for x in range(5)]
    predictions = [pd.read_csv(os.path.join(output_dir, f"lazypredict_preds_{x}.csv")) for x in range(5)]

    pred_values = []
    cv_scores = []

    for i in range(len(models)):
        print(models[i].head())
        best_model = models[i].loc[0, "Model"]
        pred_values.append(predictions[i][best_model].values)
        cv_scores.append(models[i].loc[0, "wauc"])

    cm = np.zeros((len(pred_values), len(pred_values)))
    for i in range(len(pred_values)):
        for j in range(len(pred_values)):
            cm[i, j] = spearmanr(pred_values[i], pred_values[j]).correlation

    print(cm)

    df = defaultdict(list)
    df["Id"] = image_id
    df["Label"] = np.mean(pred_values)

    df = pd.DataFrame.from_dict(df)
    df.to_csv(os.path.join(output_dir, f"lazypredict_top1_{np.mean(cv_scores):.4f}.csv"), index=False)

    # Depth 3
    pred_values = []
    cv_scores = []

    for i in range(len(models)):
        print(models[i].head())
        for j in range(3):
            pred_values.append(predictions[i][models[i].loc[j, "Model"][0]].values)
            cv_scores.append(models[i].loc[j, "wauc"])

    df = defaultdict(list)
    df["Id"] = image_id
    df["Label"] = np.mean(pred_values)

    df = pd.DataFrame.from_dict(df)
    df.to_csv(os.path.join(output_dir, f"lazypredict_top3_{np.mean(cv_scores):.4f}.csv"), index=False)


if __name__ == "__main__":
    main()
