from typing import Union, Tuple

import numpy as np
from pytorch_toolbelt.modules import *

from alaska2.dataset import *

__all__ = ["rgb_resnet34", "rgb_resnet18", "rgb_b0", "rgb_seresnext50", "rgb_densenet121", "rgb_hrnet18"]


class RGBModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes,
        mean: Union[Tuple, List] = [0.3914976, 0.44266784, 0.46043398],
        std: Union[Tuple, List] = [0.17819773, 0.17319807, 0.18128773],
        dropout=0,
    ):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.encoder = encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(self.encoder.channels[-1], num_classes)
        self.flag_classifier = nn.Linear(self.encoder.channels[-1], 1)

    def forward(self, **kwargs):
        x = kwargs[INPUT_IMAGE_KEY]
        x = self.rgb_bn(x)
        x = self.encoder(x)
        x = self.pool(x[-1])

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY]


def rgb_b0(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = EfficientNetB0Encoder()
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout)


def rgb_resnet18(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = Resnet18Encoder(pretrained=pretrained)
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout)


def rgb_resnet34(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = Resnet34Encoder(pretrained=pretrained)
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout)


def rgb_seresnext50(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = SEResNeXt50Encoder(pretrained=pretrained)
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout)


def rgb_densenet121(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = DenseNet121Encoder(pretrained=pretrained)
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout)


def rgb_densenet201(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = DenseNet201Encoder(pretrained=pretrained)
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout)


def rgb_hrnet18(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = HRNetV2Encoder18(pretrained=pretrained)
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
