from collections import OrderedDict

import torch
from pytorch_toolbelt.modules import *
from torch import nn

from alaska2.dataset import *
import numpy as np
import torch.nn.functional as F

__all__ = ["dct_resnet34", "dct_seresnext50"]


class DCTModel(nn.Module):
    def __init__(self, dct_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        self.dct_y_stem = nn.Sequential(
            nn.BatchNorm2d(64), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.dct_cr_stem = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dct_cb_stem = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dct_encoder = dct_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.dropout = nn.Dropout(dropout)

        self.type_classifier = nn.Linear(dct_encoder.channels[-1], num_classes)
        self.flag_classifier = nn.Linear(dct_encoder.channels[-1], 1)

    def forward(self, **kwargs):
        dct_y = self.dct_y_stem(kwargs[INPUT_FEATURES_DCT_Y_KEY])
        dct_cr = self.dct_cr_stem(kwargs[INPUT_FEATURES_DCT_CR_KEY])
        dct_cb = self.dct_cb_stem(kwargs[INPUT_FEATURES_DCT_CB_KEY])

        x = torch.cat([dct_y, dct_cr, dct_cb], dim=1)
        x = self.dct_encoder(x)
        x = self.pool(x[-1])

        return {
            OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.dropout(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.dropout(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_DCT_Y_KEY, INPUT_FEATURES_DCT_CR_KEY, INPUT_FEATURES_DCT_CB_KEY]


def dct_resnet34(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = Resnet34Encoder(pretrained=pretrained).change_input_channels(64 * 3)
    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)


def dct_seresnext50(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = SEResNeXt50Encoder(pretrained=pretrained)

    layer0_modules = [
        ("conv1", nn.Conv2d(64 * 3, 64, 3, stride=2, padding=1, bias=False)),
        ("bn1", nn.BatchNorm2d(64)),
        ("relu1", nn.ReLU(inplace=True)),
        ("conv2", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
        ("bn2", nn.BatchNorm2d(64)),
        ("relu2", nn.ReLU(inplace=True)),
        ("conv3", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
        ("bn3", nn.BatchNorm2d(64)),
        ("relu3", nn.ReLU(inplace=True)),
        ("pool", nn.MaxPool2d(3, stride=2, ceil_mode=True)),
    ]

    dct_encoder.layer0 = nn.Sequential(OrderedDict(layer0_modules))
    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)
