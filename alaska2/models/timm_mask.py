import numpy as np
from pytorch_toolbelt.modules import Normalize
from torch import nn

from alaska2.dataset import (
    INPUT_IMAGE_KEY,
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    OUTPUT_PRED_MODIFICATION_MASK,
)

__all__ = ["TimmRgbMaskModel", "nr_rgb_tf_efficientnet_b3_ns_mish_mask"]


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


# Model zoo


def nr_rgb_tf_efficientnet_b3_ns_mish_mask(num_classes=4, pretrained=True, dropout=0.2):
    from timm.models.layers import Mish
    from .timm import patched_tf_efficientnet_b3_ns

    encoder = patched_tf_efficientnet_b3_ns(pretrained=pretrained, act_layer=Mish, path_drop_rate=0.2)
    del encoder.classifier

    return TimmRgbMaskModel(encoder, num_classes=num_classes, dropout=dropout)
