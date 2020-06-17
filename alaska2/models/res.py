import numpy as np
import torch
from pytorch_toolbelt.modules import Normalize, GlobalAvgPool2d
from pytorch_toolbelt.utils import transfer_weights
from timm.models import efficientnet
from torch import nn

from alaska2.dataset import (
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    INPUT_IMAGE_KEY,
    INPUT_FEATURES_DECODING_RESIDUAL_KEY,
)

__all__ = ["res_tf_efficientnet_b2_ns", "rgb_res_tf_efficientnet_b2_ns", "rgb_res_sms_tf_efficientnet_b2_ns"]


class ResidualOnlyModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
    ):
        super().__init__()
        self.res_bn = Normalize([2, 2, 2], [1, 1, 1])
        self.encoder = encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        x = self.res_bn(kwargs[INPUT_FEATURES_DECODING_RESIDUAL_KEY])
        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            # OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_DECODING_RESIDUAL_KEY]


class ImageAndResidualModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
    ):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.res_bn = Normalize([2, 2, 2], [1, 1, 1])
        self.encoder = encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        rgb = self.rgb_bn(kwargs[INPUT_IMAGE_KEY])
        res = self.res_bn(kwargs[INPUT_FEATURES_DECODING_RESIDUAL_KEY])
        x = torch.cat([rgb, res], dim=1)
        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_DECODING_RESIDUAL_KEY]


class SiameseImageAndResidualModel(nn.Module):
    def __init__(
        self,
        rgb_encoder,
        res_encoder,
        num_classes,
        dropout=0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
    ):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.res_bn = Normalize([2, 2, 2], [1, 1, 1])
        self.rgb_encoder = rgb_encoder
        self.res_encoder = res_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(rgb_encoder.num_features + res_encoder.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.type_classifier = nn.Linear(512, num_classes)
        self.flag_classifier = nn.Linear(512, 1)

    def forward(self, **kwargs):
        rgb = self.rgb_encoder.forward_features(self.rgb_bn(kwargs[INPUT_IMAGE_KEY]))
        rgb = self.pool(rgb)

        res = self.res_encoder.forward_features(self.res_bn(kwargs[INPUT_FEATURES_DECODING_RESIDUAL_KEY]))
        res = self.pool(res)

        x = torch.cat([rgb, res], dim=1)
        x = self.decoder(x)

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_DECODING_RESIDUAL_KEY]


def res_tf_efficientnet_b2_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained, drop_path_rate=0.1)
    del encoder.classifier

    return ResidualOnlyModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_res_tf_efficientnet_b2_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b2_ns(in_chans=6, pretrained=False, drop_path_rate=0.1)
    del encoder.classifier

    if pretrained:
        donor = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained)
        transfer_weights(encoder, donor.state_dict())

    return ImageAndResidualModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_res_sms_tf_efficientnet_b2_ns(num_classes=4, pretrained=True, dropout=0):
    rgb_encoder = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained, drop_path_rate=0.1)
    del rgb_encoder.classifier

    res_encoder = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained, drop_path_rate=0.1)
    del res_encoder.classifier

    return SiameseImageAndResidualModel(
        rgb_encoder,
        res_encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=rgb_encoder.default_cfg["mean"],
        std=rgb_encoder.default_cfg["std"],
    )
