from collections import OrderedDict

import torch
from pytorch_toolbelt.modules import *
from timm.models import skresnext50_32x4d
from torch import nn

from alaska2.dataset import *
import numpy as np
import torch.nn.functional as F

__all__ = ["ycrcb_skresnext50_32x4d"]


class YCrCbModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0):
        super().__init__()
        self.encoder = encoder
        self.y_conv = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=2)
        self.cr_conv = nn.Conv2d(1, 8, kernel_size=1, padding=0, stride=1)
        self.cb_conv = nn.Conv2d(1, 8, kernel_size=1, padding=0, stride=1)
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        x = torch.cat(
            [
                self.y_conv(kwargs[INPUT_FEATURES_CHANNEL_Y_KEY]),
                self.cr_conv(kwargs[INPUT_FEATURES_CHANNEL_CR_KEY]),
                self.cb_conv(kwargs[INPUT_FEATURES_CHANNEL_CB_KEY]),
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
        return [INPUT_FEATURES_CHANNEL_Y_KEY, INPUT_FEATURES_CHANNEL_CR_KEY, INPUT_FEATURES_CHANNEL_CB_KEY]


def ycrcb_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(pretrained=pretrained)
    del encoder.fc
    encoder.conv1 = make_n_channel_input(encoder.conv1, 8 * 3, "auto")

    return YCrCbModel(encoder, num_classes=num_classes, dropout=dropout)
