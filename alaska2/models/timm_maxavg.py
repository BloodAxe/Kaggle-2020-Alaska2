import numpy as np
from pytorch_toolbelt.modules import Normalize
from timm.models import efficientnet
from timm.models.layers import SelectAdaptivePool2d
from torch import nn

from alaska2.dataset import INPUT_IMAGE_KEY, OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_MODIFICATION_TYPE
from alaska2.models.classifiers import WeightNormClassifier

__all__ = ["rgb_tf_efficientnet_b2_ns_avgmax"]


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


## Cherry-picked & modified functions from TIMM/EfficientNet to use Mish


# Model zoo


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
