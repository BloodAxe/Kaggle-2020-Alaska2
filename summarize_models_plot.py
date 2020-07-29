import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    summarize = pd.read_csv("summarize.csv")

    summarize["best_auc"] = np.maximum(summarize["c_auc"].values, summarize["b_auc"].values)
    summarize = summarize.sort_values(by="best_auc")

    plt.figure(figsize=(16, 12))
    plt.scatter(
        x=summarize["model_name"],
        y=summarize["best_auc"],
        s=summarize["params_count"] * 5e-5,
        c=LabelEncoder().fit_transform(summarize["input"]),
        linewidths=1,
        alpha=0.5,
        edgecolors="black",
    )

    plt.ylabel("wAUC")
    plt.xticks(rotation=90)
    # plt.legend()
    plt.title("wAUC for different models and input")
    plt.tight_layout()
    plt.savefig(fname="wauc_summary.png")
    plt.show()


if __name__ == "__main__":
    main()
