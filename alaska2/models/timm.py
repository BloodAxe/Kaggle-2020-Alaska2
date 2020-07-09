import torch
import torch.nn.functional as F
from pytorch_toolbelt.modules import Normalize, GlobalAvgPool2d
from timm.models import skresnext50_32x4d
from timm.models import tresnet, efficientnet, resnet
from timm.models.layers import SelectAdaptivePool2d
from torch import nn
import numpy as np

from alaska2.models.classifiers import WeightNormClassifier

from alaska2.dataset import (
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    INPUT_IMAGE_KEY,
    INPUT_IMAGE_QF_KEY,
    INPUT_FEATURES_JPEG_FLOAT,
    OUTPUT_PRED_MODIFICATION_MASK,
)

__all__ = [
    "rgb_skresnext50_32x4d",
    "rgb_tf_efficientnet_b2_ns_avgmax",
    "rgb_tf_efficientnet_b6_ns",
    "rgb_tf_efficientnet_b1_ns",
    "rgb_swsl_resnext101_32x8d",
    "rgb_tf_efficientnet_b2_ns",
    "rgb_tf_efficientnet_b3_ns",
    "rgb_tresnet_m_448",
    "rgb_qf_tf_efficientnet_b2_ns",
    "rgb_qf_tf_efficientnet_b6_ns",
    "rgb_qf_swsl_resnext101_32x8d",
    "rgb_tf_efficientnet_b7_ns",
    # Models using unrounded image
    "nr_rgb_tf_efficientnet_b3_ns_mish",
    "nr_rgb_tf_efficientnet_b3_ns_mish_mask",
    "nr_rgb_tf_efficientnet_b6_ns",
    "nr_rgb_mixnet_xl",
    "nr_rgb_mixnet_xxl",
    "nr_rgb_tf_efficientnet_b3_ns_gn"
]


class TimmRgbModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
        max_pixel_value=255,
        input_key=INPUT_IMAGE_KEY,
    ):
        super().__init__()
        self.encoder = encoder

        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)
        self.input_key = input_key

    def forward(self, **kwargs):
        x = kwargs[self.input_key]
        x = self.rgb_bn(x)
        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [self.input_key]


class TimmRgbMaskModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
        max_pixel_value=255,
        input_key=INPUT_IMAGE_KEY,
    ):
        super().__init__()
        self.encoder = encoder

        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.drop = nn.Dropout2d(dropout)

        # self.upsample = nn.Sequential(
        #     nn.Conv2d(encoder.num_features, 512, kernel_size=1),
        #     nn.PixelShuffle(upscale_factor=2),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=1),
        #     nn.PixelShuffle(upscale_factor=2),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32,1,kernel_size=1)
        # )

        self.mask = nn.Sequential(
            nn.Conv2d(encoder.num_features, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

        self.type_classifier = nn.Conv2d(encoder.num_features, num_classes, kernel_size=1)
        self.flag_classifier = nn.Conv2d(encoder.num_features, 1, kernel_size=1)
        self.input_key = input_key

    def forward(self, **kwargs):
        x = kwargs[self.input_key]
        x = self.rgb_bn(x)
        x = self.encoder.forward_features(x)

        x = self.drop(x)
        mask = self.mask(x)
        mask_s = mask.sigmoid()
        flag = self.flag_classifier(x) * mask_s
        type = self.type_classifier(x) * mask_s

        return {
            OUTPUT_PRED_MODIFICATION_MASK: mask,
            OUTPUT_PRED_MODIFICATION_FLAG: flag.mean(dim=(2, 3)),
            OUTPUT_PRED_MODIFICATION_TYPE: type.mean(dim=(2, 3)),
        }

    @property
    def required_features(self):
        return [self.input_key]


class TimmRgbModelAvgMax(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
    ):
        super().__init__()
        self.encoder = encoder
        max_pixel_value = 255
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.pool = SelectAdaptivePool2d(pool_type="catavgmax", flatten=True)
        self.type_classifier = WeightNormClassifier(encoder.num_features * 2, num_classes, 128, dropout=dropout)
        self.flag_classifier = WeightNormClassifier(encoder.num_features * 2, 1, 128, dropout=dropout)

    def forward(self, **kwargs):
        x = kwargs[INPUT_IMAGE_KEY]
        x = self.rgb_bn(x.float())
        x = self.encoder.forward_features(x)
        x = self.pool(x)
        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY]


class ImageAndQFModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
    ):
        super().__init__()
        self.encoder = encoder
        max_pixel_value = 255
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)

        # Recombination of embedding and quality factor
        self.fc1 = nn.Sequential(nn.Linear(encoder.num_features + 3, encoder.num_features), nn.ReLU())

        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        x = kwargs[INPUT_IMAGE_KEY]
        qf = F.one_hot(kwargs[INPUT_IMAGE_QF_KEY], 3)

        x = self.rgb_bn(x.float())
        x = self.encoder.forward_features(x)
        x = self.pool(x)

        x = torch.cat([x, qf.type_as(x)], dim=1)
        x = self.fc1(x)

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_IMAGE_QF_KEY]


def rgb_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(pretrained=pretrained)
    del encoder.fc

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_tresnet_m_448(num_classes=4, pretrained=True, dropout=0):
    encoder = tresnet.tresnet_m_448(pretrained=pretrained)
    del encoder.head

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_swsl_resnext101_32x8d(num_classes=4, pretrained=True, dropout=0):
    encoder = resnet.swsl_resnext101_32x8d(pretrained=pretrained)
    del encoder.fc

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_tf_efficientnet_b2_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_tf_efficientnet_b3_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=pretrained)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_tf_efficientnet_b1_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b1_ns(pretrained=pretrained)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_tf_efficientnet_b2_ns_avgmax(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained)
    del encoder.classifier

    return TimmRgbModelAvgMax(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_tf_efficientnet_b6_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b6_ns(pretrained=pretrained)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def nr_rgb_tf_efficientnet_b3_ns_mish(num_classes=4, pretrained=True, dropout=0):
    from timm.models.layers import Mish

    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=pretrained, act_layer=Mish)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def nr_rgb_tf_efficientnet_b3_ns_gn(num_classes=4, pretrained=True, dropout=0):
    from timm.models.layers import Mish

    def group_norm_builder(channels):
        groups = [32, 24, 16, 8, 7, 6, 5, 4, 3, 2]
        for g in groups:
            if channels % g == 0 and channels > g:
                norm_layer = nn.GroupNorm(g, channels)
                norm_layer.weight.data.fill_(1.)
                norm_layer.bias.data.zero_()
                print(f"Created nn.GroupNorm(groups={g}, channels={channels})")
                return norm_layer

    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=pretrained, norm_layer=group_norm_builder)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def nr_rgb_tf_efficientnet_b3_ns_mish_mask(num_classes=4, pretrained=True, dropout=0):
    from timm.models.layers import Mish

    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=pretrained, act_layer=Mish)
    del encoder.classifier

    return TimmRgbMaskModel(encoder, num_classes=num_classes, dropout=dropout)


def nr_rgb_tf_efficientnet_b6_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b6_ns(pretrained=pretrained)
    del encoder.classifier
    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout, input_key=INPUT_FEATURES_JPEG_FLOAT)


def nr_rgb_mixnet_xl(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.mixnet_xl(pretrained=pretrained)
    del encoder.classifier
    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout, input_key=INPUT_FEATURES_JPEG_FLOAT)


def nr_rgb_mixnet_xxl(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.mixnet_xxl(pretrained=pretrained)
    del encoder.classifier
    mean = encoder.default_cfg["mean"]
    std = encoder.default_cfg["std"]

    if pretrained:
        from torch.utils import model_zoo

        src_state_dict = model_zoo.load_url(
            "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth"
        )
        dst_state_dict = encoder.state_dict()

        for key, dst_tensor in dst_state_dict.items():
            dst_tensor_size = dst_tensor.size()
            if key in src_state_dict:
                src_tensor = src_state_dict[key]
                src_tensor_size = src_tensor.size()
                # If shape of tensors does not match, we pad necessary with random weights
                if src_tensor_size != dst_tensor_size:
                    assert len(src_tensor_size) == len(dst_tensor_size)
                    old = src_state_dict[key]
                    src_state_dict[key] = dst_tensor.clone()
                    slice_size = [slice(0, src_size) for src_size, dst_size in zip(src_tensor_size, dst_tensor_size)]
                    src_state_dict[key][slice_size] = old

        encoder.load_state_dict(src_state_dict, strict=False)

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout, input_key=INPUT_FEATURES_JPEG_FLOAT)


def rgb_tf_efficientnet_b7_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b7_ns(pretrained=pretrained, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


# RGB + QF
def rgb_qf_tf_efficientnet_b2_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained)
    del encoder.classifier

    return ImageAndQFModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_qf_tf_efficientnet_b6_ns(num_classes=4, pretrained=True, dropout=0):
    encoder = efficientnet.tf_efficientnet_b6_ns(pretrained=pretrained)
    del encoder.classifier

    return ImageAndQFModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_qf_swsl_resnext101_32x8d(num_classes=4, pretrained=True, dropout=0):
    encoder = resnet.swsl_resnext101_32x8d(pretrained=pretrained)
    del encoder.fc

    return ImageAndQFModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )
