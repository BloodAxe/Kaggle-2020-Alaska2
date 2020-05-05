from pytorch_toolbelt.modules import *
from alaska2.dataset import *
from .srnet import Srnet

__all__ = ["rgb_dct_efficientb3", "rgb_dct_resnet34", "rgb_dct_b0_srnet", "rgb_dct_seresnext50"]


class RGBDCTSiamese(nn.Module):
    def __init__(self, rgb_encoder: EncoderModule, dct_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        self.rgb_bn = nn.BatchNorm2d(3)
        self.dct_bn = nn.BatchNorm2d(64)
        self.rgb_encoder = rgb_encoder
        self.dct_encoder = dct_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.embedding = nn.Sequential(
            nn.Linear(self.rgb_encoder.channels[-1] + self.dct_encoder.channels[-1], 128),
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
        rgb = self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float())
        dct = self.dct_bn(kwargs[INPUT_DCT_KEY].float())

        rgb_features = self.pool(self.rgb_encoder(rgb)[-1])
        dct_featues = self.pool(self.dct_encoder(dct)[-1])

        x = torch.cat([rgb_features, dct_featues], dim=1)
        x = self.embedding(x)

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }


def rgb_dct_efficientb3(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = EfficientNetB3Encoder()
    dct_encoder = EfficientNetB3Encoder().change_input_channels(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)


def rgb_dct_resnet34(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = Resnet34Encoder()
    dct_encoder = Resnet34Encoder().change_input_channels(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)


def rgb_dct_b0_srnet(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = EfficientNetB0Encoder()
    dct_encoder = Srnet(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)


def rgb_dct_seresnext50(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = SEResNeXt50Encoder()
    dct_encoder = SEResNeXt50Encoder().change_input_channels(64)
    return RGBDCTSiamese(rgb_encoder, dct_encoder, num_classes=num_classes, dropout=dropout)
