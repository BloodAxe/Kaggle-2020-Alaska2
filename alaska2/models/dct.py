from collections import OrderedDict

import torch
from pytorch_toolbelt.modules import *
from timm.models import skresnext50_32x4d
from torch import nn

from alaska2.dataset import *
import numpy as np
import torch.nn.functional as F

__all__ = ["dct_skresnext50_32x4d"]


class DCTModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0):
        super().__init__()
        self.encoder = encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        x = torch.cat(
            [
                kwargs[INPUT_FEATURES_DCT_Y_KEY],
                F.interpolate(kwargs[INPUT_FEATURES_DCT_CR_KEY], scale_factor=2, mode="nearest"),
                F.interpolate(kwargs[INPUT_FEATURES_DCT_CB_KEY], scale_factor=2, mode="nearest"),
            ],
            dim=1,
        )

        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_DCT_Y_KEY, INPUT_FEATURES_DCT_CR_KEY, INPUT_FEATURES_DCT_CB_KEY]


def dct_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(pretrained=pretrained)
    del encoder.fc

    return DCTModel(encoder, num_classes=num_classes, dropout=dropout)
