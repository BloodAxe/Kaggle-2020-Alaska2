from collections import OrderedDict

import torch
from pytorch_toolbelt.modules import *
from timm.models import skresnext50_32x4d
from torch import nn

from alaska2.dataset import *
import numpy as np
import torch.nn.functional as F

__all__ = ["ycrcb_skresnext50_32x4d"]


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class YCrCbModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0):
        super().__init__()
        self.rgb_encoder = encoder
        self.norm = Normalize([-10.5957038, -3.62235547, 2.02056952], [42.37946293, 8.89775623, 8.94904454])

        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        y = kwargs[INPUT_FEATURES_CHANNEL_Y_KEY]
        cb = kwargs[INPUT_FEATURES_CHANNEL_CB_KEY]
        cr = kwargs[INPUT_FEATURES_CHANNEL_CR_KEY]

        # print('y', y.mean().item(), y.std().item())
        # print('cr', cr.mean().item(), cr.std().item())
        # print('cb', cb.mean().item(), cb.std().item())

        x = torch.cat([y, cb, cr], dim=1)
        x = self.norm(x)
        x = self.rgb_encoder.forward_features(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_CHANNEL_Y_KEY, INPUT_FEATURES_CHANNEL_CR_KEY, INPUT_FEATURES_CHANNEL_CB_KEY]


class HalvedYCrCbModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0):
        super().__init__()
        self.rgb_encoder = encoder
        # YCrCB (array([-15.25726164,  -7.29664547,   3.33329585]), array([43.31931419, 10.97596226, 10.13833837]))
        self.y_norm = Normalize()
        self.cb_norm = Normalize()
        self.cr_norm = Normalize()

        self.s2d = SpaceToDepth(block_size=2)

        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        y = self.y_norm(kwargs[INPUT_FEATURES_CHANNEL_Y_KEY])
        cb = self.cb_norm(kwargs[INPUT_FEATURES_CHANNEL_CB_KEY])
        cr = self.cr_norm(kwargs[INPUT_FEATURES_CHANNEL_CR_KEY])
        y = self.d2s(y)

        x = torch.cat([y, cb, cr], dim=1)
        x = self.norm(x)
        x = self.rgb_encoder.forward_features(x)
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
    # encoder.conv1 = make_n_channel_input(encoder.conv1, 6, "auto")

    return YCrCbModel(encoder, num_classes=num_classes, dropout=dropout)


def ycrcb_halved_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(pretrained=pretrained)
    del encoder.fc
    encoder.conv1 = make_n_channel_input(encoder.conv1, 6, "auto")

    return HalvedYCrCbModel(encoder, num_classes=num_classes, dropout=dropout)
