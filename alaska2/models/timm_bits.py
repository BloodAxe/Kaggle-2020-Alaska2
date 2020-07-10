from collections import OrderedDict

from timm.models.layers import Mish
from torch import nn
from torch.nn import InstanceNorm2d

from alaska2.dataset import (
    INPUT_IMAGE_KEY,
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    OUTPUT_PRED_PAYLOAD_BITS,
    INPUT_FEATURES_JPEG_FLOAT,
)

__all__ = [
    "nr_rgb_tf_efficientnet_b3_ns_mish_bits",
    "nr_rgb_tf_efficientnet_b3_ns_in_mish_bits",
]

from alaska2.models.timm import TimmRgbModel, patched_tf_efficientnet_b3_ns


class TimmRgbModelBits(TimmRgbModel):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0.0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
        max_pixel_value=255,
        input_key=INPUT_IMAGE_KEY,
    ):
        super().__init__(encoder, num_classes, dropout, mean, std, max_pixel_value, input_key)
        self.bits = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(encoder.num_features, 128, kernel_size=1)),
                    ("act1", Mish()),
                    ("drop", nn.Dropout2d(0.2, inplace=True)),
                    ("conv2", nn.Conv2d(128, 1, kernel_size=1)),
                    ("relu", nn.ReLU(True)),
                ]
            )
        )

    def forward(self, **kwargs):
        x = kwargs[self.input_key]
        x = self.rgb_bn(x)
        last_fm = self.encoder.forward_features(x)
        x = self.pool(last_fm)
        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
            OUTPUT_PRED_PAYLOAD_BITS: self.bits_regression(last_fm).sum(),
        }

    @property
    def required_features(self):
        return [self.input_key]


def nr_rgb_tf_efficientnet_b3_ns_mish_bits(num_classes=4, pretrained=True, dropout=0.2):
    encoder = patched_tf_efficientnet_b3_ns(pretrained=pretrained, act_layer=Mish, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModelBits(encoder, num_classes=num_classes, dropout=dropout, input_key=INPUT_FEATURES_JPEG_FLOAT)


def nr_rgb_tf_efficientnet_b3_ns_in_mish_bits(num_classes=4, pretrained=True, dropout=0.2):
    def instance_norm_builder(channels, **kwargs):
        norm_layer = InstanceNorm2d(channels, affine=True)
        return nn.Sequential(OrderedDict([("in", norm_layer)]))

    encoder = patched_tf_efficientnet_b3_ns(
        pretrained=pretrained, drop_path_rate=0.2, norm_layer=instance_norm_builder, act_layer=Mish
    )
    print(encoder)
    del encoder.classifier

    return TimmRgbModelBits(encoder, num_classes=num_classes, dropout=dropout, input_key=INPUT_FEATURES_JPEG_FLOAT)
