import torch
from pytorch_toolbelt.modules import Normalize, GlobalAvgPool2d
from pytorch_toolbelt.modules.activations import Mish
from pytorch_toolbelt.utils import transfer_weights, fs
from timm.models import skresnext50_32x4d, tresnet, resnet, res2net, efficientnet
from timm.models import dpn

from torch import nn
import numpy as np

from alaska2.dataset import (
    OUTPUT_PRED_EMBEDDING,
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    INPUT_IMAGE_KEY,
    INPUT_FEATURES_ELA_KEY,
    INPUT_FEATURES_ELA_RICH_KEY,
)

__all__ = [
    "ela_tf_efficientnet_b2_ns",
    "ela_tf_efficientnet_b6_ns",
    "ela_skresnext50_32x4d",
    "ela_rich_skresnext50_32x4d",
    "ela_wider_resnet38",
    "ela_ecaresnext26tn_32x4d",
]


class TimmRgbElaModel(nn.Module):
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
        self.ela_bn = Normalize([0, 0, 0], [0.2 * 127, 0.2 * 127, 0.2 * 127])
        self.encoder = encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        rgb = self.rgb_bn(kwargs[INPUT_IMAGE_KEY])
        ela = self.ela_bn(kwargs[INPUT_FEATURES_ELA_KEY])
        x = torch.cat([rgb, ela], dim=1)
        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            # OUTPUT_PRED_EMBEDDING: x,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_ELA_KEY]


class TimmRgbElaRichModel(nn.Module):
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
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_ELA_RICH_KEY]


def ela_tf_efficientnet_b2_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b2_ns(in_chans=6, pretrained=False, drop_path_rate=0.1)
    del encoder.classifier

    if pretrained:
        donor = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained)
        transfer_weights(encoder, donor.state_dict())

    return TimmRgbElaModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def ela_tf_efficientnet_b6_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b6_ns(in_chans=6, pretrained=False, drop_path_rate=0.2)
    del encoder.classifier

    if pretrained:
        donor = efficientnet.tf_efficientnet_b6_ns(pretrained=pretrained)
        transfer_weights(encoder, donor.state_dict())

    return TimmRgbElaModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def ela_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(stem_type="deep", in_chans=6)
    del encoder.fc

    if pretrained:
        donor = skresnext50_32x4d(pretrained=True)
        transfer_weights(encoder, donor.state_dict())

    return TimmRgbElaModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_rich_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(stem_type="deep", in_chans=7)
    del encoder.fc

    if pretrained:
        donor = skresnext50_32x4d(pretrained=True)
        transfer_weights(encoder, donor.state_dict())

    return TimmRgbElaRichModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_tresnet_m(num_classes=4, pretrained=True, dropout=0):
    encoder = tresnet.tresnet_m(pretrained=pretrained)
    del encoder.fc

    return TimmRgbElaModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_ecaresnext26tn_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = resnet.ecaresnext26tn_32x4d(in_chans=6)
    del encoder.fc

    if pretrained:
        donor = resnet.ecaresnext26tn_32x4d(pretrained=True)
        transfer_weights(encoder, donor.state_dict())

    return TimmRgbElaModel(encoder, num_classes=num_classes, dropout=dropout)


def ela_wider_resnet38(num_classes=4, pretrained=True, dropout=0):
    from alaska2.models.backbones.wider_resnet import wider_resnet38

    encoder = wider_resnet38(in_chans=6)
    if pretrained:
        checkpoint = torch.load(fs.auto_file("wide_resnet38_ipabn_lr_256.pth.tar"), map_location="cpu")
        transfer_weights(encoder, checkpoint["state_dict"])
        print("Loaded weights from Mapilary")

    return TimmRgbElaModel(encoder, num_classes=num_classes, dropout=dropout)
