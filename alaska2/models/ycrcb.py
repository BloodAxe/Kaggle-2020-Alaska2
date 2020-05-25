from collections import OrderedDict

import torch
from pytorch_toolbelt.modules import *
from timm.models import skresnext50_32x4d
from torch import nn

from alaska2.dataset import *
import numpy as np
import torch.nn.functional as F

__all__ = ["ycrcb_skresnext50_32x4d", "ela_s2d_skresnext50_32x4d"]


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
        self.ycbcr_encoder = encoder
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


class ELAYCbCcS2DModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0, abn_block=ABN):
        super().__init__()
        self.encoder = encoder
        self.norm = Normalize([-10.5957038, -3.62235547, 2.02056952], [42.37946293, 8.89775623, 8.94904454])
        block_size = 4
        self.s2d = SpaceToDepth(block_size=block_size)
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d((3 + 3) * (block_size ** 2), 96, kernel_size=1)),
                    ("abn1", abn_block(96)),
                    ("conv2", nn.Conv2d(96, 64, kernel_size=1)),
                ]
            )
        )

        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):

        ycbcr = torch.cat(
            [
                kwargs[INPUT_FEATURES_CHANNEL_Y_KEY],
                kwargs[INPUT_FEATURES_CHANNEL_CB_KEY],
                kwargs[INPUT_FEATURES_CHANNEL_CR_KEY],
            ],
            dim=1,
        )
        ycbcr = self.norm(ycbcr)

        ela = kwargs[INPUT_FEATURES_ELA_RICH_KEY]
        x = torch.cat([ycbcr, ela], dim=1)

        x = self.s2d(x)
        x = self.conv1(x)

        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [
            INPUT_FEATURES_CHANNEL_Y_KEY,
            INPUT_FEATURES_CHANNEL_CR_KEY,
            INPUT_FEATURES_CHANNEL_CB_KEY,
            INPUT_FEATURES_ELA_RICH_KEY,
        ]


def ycrcb_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(stem_type="deep", in_chans=6)
    del encoder.fc
    # encoder.conv1 = make_n_channel_input(encoder.conv1, 6, "auto")

    return YCrCbModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_s2d_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(stem_type="deep", in_chans=64)
    del encoder.fc

    return ELAYCbCcS2DModel(encoder, num_classes=num_classes, dropout=dropout)
