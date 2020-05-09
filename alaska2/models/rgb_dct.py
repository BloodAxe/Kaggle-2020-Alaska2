from pytorch_toolbelt.modules import *
from alaska2.dataset import *
from .srnet import Srnet

__all__ = ["rgb_dct_efficientb3", "rgb_dct_resnet34", "rgb_dct_b0_srnet", "rgb_dct_seresnext50"]


class RGBDCTSiamese(nn.Module):
    def __init__(self, rgb_encoder: EncoderModule, dct_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(
            [0.3914976 * max_pixel_value, 0.44266784 * max_pixel_value, 0.46043398 * max_pixel_value],
            [0.17819773 * max_pixel_value, 0.17319807 * max_pixel_value, 0.18128773 * max_pixel_value],
        )
        self.rgb_encoder = rgb_encoder

        self.dct_bn = nn.BatchNorm2d(64)
        self.dct_encoder = dct_encoder

        self.pool = GlobalAvgPool2d(flatten=True)
        self.embedding = nn.Sequential(
            nn.Linear(self.rgb_encoder.channels[-1] + self.dct_encoder.channels[-1], 128),
            nn.AlphaDropout(dropout),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

        self.type_classifier = nn.Linear(128, num_classes)
        self.flag_classifier = nn.Linear(128, 1)

    def forward(self, **kwargs):
        rgb_image = self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float())
        dct_image = self.dct_bn(kwargs[INPUT_FEATURES_DCT_KEY])

        rgb_features = self.rgb_encoder(rgb_image)
        dct_features = self.dct_encoder(dct_image)

        rgb_features = self.pool(rgb_features[-1])
        dct_features = self.pool(dct_features[-1])

        x = torch.cat([rgb_features, dct_features], dim=1)
        embedding = self.embedding(x)

        return {
            OUTPUT_PRED_EMBEDDING: embedding,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(embedding),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(embedding),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_DCT_KEY]


def rgb_dct_efficientb3(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = EfficientNetB3Encoder()
    dct_encoder = EfficientNetB3Encoder().change_input_channels(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)


def rgb_dct_resnet34(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = Resnet34Encoder(pretrained=pretrained)
    dct_encoder = Resnet34Encoder(pretrained=pretrained).change_input_channels(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)


def rgb_dct_b0_srnet(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = EfficientNetB0Encoder()
    dct_encoder = Srnet(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)


def rgb_dct_seresnext50(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = SEResNeXt50Encoder()
    dct_encoder = SEResNeXt50Encoder().change_input_channels(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)
