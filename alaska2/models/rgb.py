import torch
from pytorch_toolbelt.modules import *
from alaska2.dataset import *

__all__ = ["rgb_resnet34", "rgb_resnet18", "rgb_b0", "rgb_seresnext50", "rgb_densenet121", "rgb_hrnet18"]


class RGBModel(nn.Module):
    def __init__(self, rgb_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(
            [0.3914976 * max_pixel_value, 0.44266784 * max_pixel_value, 0.46043398 * max_pixel_value],
            [0.17819773 * max_pixel_value, 0.17319807 * max_pixel_value, 0.18128773 * max_pixel_value],
        )
        self.rgb_encoder = rgb_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.embedding = nn.Sequential(
            nn.Linear(self.rgb_encoder.channels[-1], 128),
            nn.BatchNorm1d(128),
            nn.AlphaDropout(dropout),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

        self.type_classifier = nn.Linear(128, num_classes)
        self.flag_classifier = nn.Linear(128, 1)

    def forward(self, **kwargs):
        image = self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float())
        features = self.rgb_encoder(image)
        embedding = self.pool(features[-1])

        x = self.embedding(embedding)

        return {
            # OUTPUT_FEATURE_MAP_4: features[0],
            # OUTPUT_FEATURE_MAP_8: features[1],
            # OUTPUT_FEATURE_MAP_16: features[2],
            # OUTPUT_FEATURE_MAP_32: features[3],
            OUTPUT_PRED_EMBEDDING: embedding,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY]


class RGBModelAllPool(nn.Module):
    def __init__(self, rgb_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(
            [0.3914976 * max_pixel_value, 0.44266784 * max_pixel_value, 0.46043398 * max_pixel_value],
            [0.17819773 * max_pixel_value, 0.17319807 * max_pixel_value, 0.18128773 * max_pixel_value],
        )
        self.rgb_encoder = rgb_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.dropout = nn.Dropout(dropout)

        features = sum(rgb_encoder.channels)
        self.type_classifier = nn.Linear(features, num_classes)
        self.flag_classifier = nn.Linear(features, 1)

    def forward(self, **kwargs):
        image = self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float())
        features = self.rgb_encoder(image)

        features = [self.pool(f) for f in features]
        features = torch.cat(features, dim=1)

        return {
            # OUTPUT_FEATURE_MAP_4: features[0],
            # OUTPUT_FEATURE_MAP_8: features[1],
            # OUTPUT_FEATURE_MAP_16: features[2],
            # OUTPUT_FEATURE_MAP_32: features[3],
            OUTPUT_PRED_EMBEDDING: features,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.dropout(features)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.dropout(features)),
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
    return RGBModelAllPool(rgb_encoder, num_classes=num_classes, dropout=dropout)
