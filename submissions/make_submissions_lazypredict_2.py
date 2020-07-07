import os
# Used to ignore warnings generated from StackingCVClassifier
import warnings
from collections import defaultdict

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd

from alaska2 import INPUT_IMAGE_KEY, get_test_dataset

# Classifiers

warnings.simplefilter("ignore")


def main():
    output_dir = os.path.dirname(__file__)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    image_id = [os.path.basename(x) for x in test_ds.images]

    models = [pd.read_csv(os.path.join(output_dir, f"lazypredict_models_{x}.csv")) for x in range(5)]
    predictions = [pd.read_csv(os.path.join(output_dir, f"lazypredict_preds_{x}.csv")) for x in range(5)]

    # for s in cv_scores:
    #     print(s)
    # print(np.mean(cv_scores), np.std(cv_scores))

    for i in range(len(models)):
        print(models[i].head())

    df = defaultdict(list)
    df["Id"] = image_id
    df["Label"] = np.mean(
        [
            predictions[0][models[0].first_valid_index].values,
            predictions[1][models[1].first_valid_index].values,
            predictions[2][models[2].first_valid_index].values,
            predictions[3][models[3].first_valid_index].values,
            predictions[4][models[4].first_valid_index].values,
        ]
    )

    cv_scores = np.mean(
        [
            predictions[0].loc[0, "wauc"],
            predictions[1].loc[0, "wauc"],
            predictions[2].loc[0, "wauc"],
            predictions[3].loc[0, "wauc"],
            predictions[4].loc[0, "wauc"],
        ]
    )

    df = pd.DataFrame.from_dict(df)
    df.to_csv(os.path.join(output_dir, f"lazypredict_{np.mean(cv_scores):.4f}.csv"), index=False)


if __name__ == "__main__":
    main()
