import torch
from torch import nn

from .rgb_dct import *

MODEL_REGISTRY = {"rgb_dct_efficientb3": rgb_dct_efficientb3, "rgb_dct_resnet34": rgb_dct_resnet34}

__all__ = ["MODEL_REGISTRY", "get_model"]


def get_model(model_name, num_classes=4, dropout=0, pretrained=True):
    return MODEL_REGISTRY[model_name](num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def model_from_checkpoint(model_checkpoint: str, model_name=None, report=True) -> nn.Module:
    checkpoint = torch.load(model_checkpoint, map_location="cpu")
    model_name = model_name or checkpoint["checkpoint_data"]["cmd_args"]["model"]

    if report:
        print(model_checkpoint, model_name, checkpoint["epoch_metrics"]["valid"]["lb"])

    model = get_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    del checkpoint

    return model.eval()
