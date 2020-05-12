from functools import partial

import pandas as pd
import torch
from pytorch_toolbelt.utils import logit


def temperature_scaling(x, t):
    x = torch.tensor(x)
    x_l = logit(x)
    x_s = torch.sigmoid(x_l * t)
    return float(x_s)


df = pd.read_csv("880_submission_type.csv")
df["Label"] = df["Label"].apply(partial(temperature_scaling, t=5))
df.to_csv("880_submission_type_temp_5.csv", index=None)

df = pd.read_csv("880_submission_type.csv")
df["Label"] = df["Label"].apply(partial(temperature_scaling, t=1 / 5))
df.to_csv("880_submission_type_temp_1_over_5.csv", index=None)
