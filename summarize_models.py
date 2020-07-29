from collections import defaultdict

from pytorch_toolbelt.utils import fs, count_parameters
import glob, os
import pandas as pd, torch

from alaska2 import get_model
from alaska2.submissions import infer_fold


def infer_activation(model_name):
    if "mish" in model_name:
        return "mish"
    if "tf_efficientnet" in model_name:
        return "swish"
    return "relu"


def infer_input(session_name):
    if session_name.startswith("rgb_"):
        return "RGB (R)"
    if session_name.startswith("rgb_qf_"):
        return "RGB+QF"
    if session_name.startswith("nr_rgb_"):
        return "RGB (NR)"
    if session_name.startswith("dct_"):
        return "DCT"
    if session_name.startswith("rgb_res_"):
        return "RGB+RES"
    if session_name.startswith("ela_"):
        return "RGB+ELA"
    return "RGB (R)"


import re
import numpy as np


def infer_model(x):
    x = fs.id_from_fname(x)
    x = x.replace("fp16", "").replace("fold", "f").replace("local_rank_0", "")
    x = re.sub(r"([ABCDEKFGHK]_)?\w{3}\d{2}_\d{2}_\d{2}", "", x).replace("_", "")
    return x


def main():
    best_checkpoints = glob.glob("runs/**/*best.pth", recursive=True)
    print(best_checkpoints)

    df = defaultdict(dict)

    for checkpoint in best_checkpoints:
        components = str.split(checkpoint, os.sep)
        session = components[1]

        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        metrics = checkpoint_data["valid_metrics"]

        b_auc = metrics.get("auc", 0)
        c_auc = metrics.get("auc_classifier", 0)
        epoch = checkpoint_data["epoch"]
        model_name = checkpoint_data["checkpoint_data"]["cmd_args"].get("model", None)
        if model_name is None:
            model_name = infer_model(session)

        df[session]["session"] = session
        df[session]["fold"] = infer_fold(session)
        df[session]["b_auc"] = max(b_auc, df[session].get("b_auc", 0))
        df[session]["c_auc"] = max(c_auc, df[session].get("c_auc", 0))
        df[session]["model_name"] = model_name
        df[session]["activation"] = infer_activation(model_name)
        df[session]["input"] = infer_input(model_name)

        if "params_count" not in df[session]:
            df[session]["params_count"] = count_parameters(get_model(model_name, pretrained=False))["total"]

        print(df[session])

    df = list(df.values())

    df.append(
        {
            "session": "May07_16_48_rgb_resnet34_fold0",
            "model_name": "rgb_resnet34",
            "fold": 0,
            "b_auc": 0.8449,
            "c_auc": np.nan,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_resnet34", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May07_16_48_rgb_resnet34_fold0",
            "model_name": "rgb_resnet34",
            "fold": 0,
            "b_auc": 0.8451,
            "c_auc": np.nan,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_resnet34", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May08_22_42_rgb_resnet34_fold1",
            "model_name": "rgb_resnet34",
            "fold": 1,
            "b_auc": 0.8439,
            "c_auc": np.nan,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_resnet34", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May09_15_13_rgb_densenet121_fold0_fp16",
            "model_name": "rgb_densenet121",
            "fold": 0,
            "b_auc": 0.8658,
            "c_auc": 0.8660,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_densenet121", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May11_08_49_rgb_densenet201_fold3_fp16",
            "model_name": "rgb_densenet201",
            "fold": 3,
            "b_auc": 0.8402,
            "c_auc": 0.8405,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_densenet201", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May13_23_00_rgb_skresnext50_32x4d_fold0_fp16",
            "model_name": "rgb_skresnext50_32x4d",
            "fold": 0,
            "b_auc": 0.9032,
            "c_auc": 0.9032,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_skresnext50_32x4d", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May13_19_06_rgb_skresnext50_32x4d_fold1_fp16",
            "model_name": "rgb_skresnext50_32x4d",
            "fold": 1,
            "b_auc": 0.9055,
            "c_auc": 0.9055,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_skresnext50_32x4d", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May12_13_01_rgb_skresnext50_32x4d_fold2_fp16",
            "model_name": "rgb_skresnext50_32x4d",
            "fold": 2,
            "b_auc": 0.9049,
            "c_auc": 0.9048,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_skresnext50_32x4d", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May11_09_46_rgb_skresnext50_32x4d_fold3_fp16",
            "model_name": "rgb_skresnext50_32x4d",
            "fold": 3,
            "b_auc": 0.8700,
            "c_auc": 0.8699,
            "activation": "relu",
            "input": "RGB",
            "params_count": count_parameters(get_model("rgb_skresnext50_32x4d", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
            "model_name": "ela_skresnext50_32x4d",
            "fold": 0,
            "b_auc": 0.9144,
            "c_auc": 0.9144,
            "activation": "relu",
            "input": "RGB+ELA",
            "params_count": count_parameters(get_model("rgb_skresnext50_32x4d", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
            "model_name": "ela_skresnext50_32x4d",
            "fold": 0,
            "b_auc": 0.9164,
            "c_auc": 0.9163,
            "activation": "relu",
            "input": "RGB+ELA",
            "params_count": count_parameters(get_model("rgb_skresnext50_32x4d", pretrained=False))["total"],
        }
    )

    df.append(
        {
            "session": "May18_20_10_ycrcb_skresnext50_32x4d_fold0_fp16",
            "model_name": "ycrcb_skresnext50_32x4d",
            "fold": 0,
            "b_auc": 0.8266,
            "c_auc": 0.8271,
            "activation": "relu",
            "input": "YCrCb",
            "params_count": count_parameters(get_model("rgb_skresnext50_32x4d", pretrained=False))["total"],
        }
    )

    df = pd.DataFrame.from_records(df)
    df.to_csv("summarize.csv", index=False)


if __name__ == "__main__":
    main()
