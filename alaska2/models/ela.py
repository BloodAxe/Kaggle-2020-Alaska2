import torch
from pytorch_toolbelt.modules import Normalize, GlobalAvgPool2d
from pytorch_toolbelt.modules.activations import Mish
from timm.models import skresnext50_32x4d
from timm.models import dpn

from torch import nn

from alaska2.dataset import (
    OUTPUT_PRED_EMBEDDING,
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    INPUT_IMAGE_KEY,
    INPUT_FEATURES_ELA_KEY,
)

__all__ = ["ela_skresnext50_32x4d"]


class TimmElaModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(
            [0.3914976 * max_pixel_value, 0.44266784 * max_pixel_value, 0.46043398 * max_pixel_value],
            [0.17819773 * max_pixel_value, 0.17319807 * max_pixel_value, 0.18128773 * max_pixel_value],
        )

        self.encoder = encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        rgb = self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float())
        ela = kwargs[INPUT_FEATURES_ELA_KEY]
        x = torch.cat([rgb, ela], dim=1)
        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_ELA_KEY]


def ela_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(stem_type="deep", in_chans=6)

    del encoder.fc

    return TimmElaModel(encoder, num_classes=num_classes, dropout=dropout)
