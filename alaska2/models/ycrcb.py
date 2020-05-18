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
        self.encoder = encoder
        # YCrCB (array([-15.25726164,  -7.29664547,   3.33329585]), array([43.31931419, 10.97596226, 10.13833837]))
        self.y_norm = Normalize([-15.25726164], [43.31931419])
        self.cr_norm = Normalize([-7.29664547], [10.97596226])
        self.cb_norm = Normalize([3.33329585], [10.13833837])

        self.y_s2d = SpaceToDepth(block_size=2)

        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        y = self.y_norm(kwargs[INPUT_FEATURES_CHANNEL_Y_KEY])
        cr = self.cr_norm(kwargs[INPUT_FEATURES_CHANNEL_CR_KEY])
        cb = self.cb_norm(kwargs[INPUT_FEATURES_CHANNEL_CB_KEY])

        x = torch.cat([self.y_s2d(y), cr, cb], dim=1)

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
    encoder.conv1 = make_n_channel_input(encoder.conv1, 6, "auto")

    return YCrCbModel(encoder, num_classes=num_classes, dropout=dropout)
