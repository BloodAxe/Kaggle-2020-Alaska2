import torch
from pytorch_toolbelt.modules import Normalize, GlobalAvgPool2d
from pytorch_toolbelt.modules.activations import Mish
from pytorch_toolbelt.utils import transfer_weights, fs
from timm.models import skresnext50_32x4d, tresnet
from timm.models import dpn

from torch import nn

from alaska2.dataset import (
    OUTPUT_PRED_EMBEDDING,
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    INPUT_IMAGE_KEY,
    INPUT_FEATURES_ELA_KEY,
    INPUT_FEATURES_ELA_RICH_KEY,
)

__all__ = ["ela_skresnext50_32x4d", "ela_rich_skresnext50_32x4d", "ela_wider_resnet38"]

from alaska2.models.backbones.wider_resnet import wider_resnet38


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


class TimmElaRichModel(nn.Module):
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
        ela = kwargs[INPUT_FEATURES_ELA_RICH_KEY]
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
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_ELA_RICH_KEY]


def ela_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(stem_type="deep", in_chans=6)
    del encoder.fc

    if pretrained:
        donor = skresnext50_32x4d(pretrained=True)
        transfer_weights(encoder, donor.state_dict())

    return TimmElaModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_rich_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(stem_type="deep", in_chans=7)
    del encoder.fc

    if pretrained:
        donor = skresnext50_32x4d(pretrained=True)
        transfer_weights(encoder, donor.state_dict())

    return TimmElaRichModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_tresnet_m(num_classes=4, pretrained=True, dropout=0):
    encoder = tresnet.tresnet_m(pretrained=pretrained)
    del encoder.fc

    return TimmElaModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_wider_resnet38(num_classes=4, pretrained=True, dropout=0):
    encoder = wider_resnet38(in_chans=6)
    if pretrained:
        checkpoint = torch.load(fs.auto_file("wide_resnet38_ipabn_lr_256.pth.tar"), map_location="cpu")
        transfer_weights(encoder, checkpoint["state_dict"])

    return TimmElaModel(encoder, num_classes=num_classes, dropout=dropout)
