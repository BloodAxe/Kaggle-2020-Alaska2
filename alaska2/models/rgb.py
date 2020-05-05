from pytorch_toolbelt.modules import *
from alaska2.dataset import *

__all__ = ["rgb_resnet34"]


class RGBModel(nn.Module):
    def __init__(self, rgb_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        self.rgb_bn = nn.BatchNorm2d(3)
        self.dct_bn = nn.BatchNorm2d(64)
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
        rgb = self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float())

        rgb_features = self.pool(self.rgb_encoder(rgb)[-1])

        x = rgb_features
        x = self.embedding(x)

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }


def rgb_resnet34(num_classes=4, dropout=0, pretrained=True):
    rgb_encoder = Resnet34Encoder(pretrained=pretrained)
    return RGBModel(rgb_encoder, num_classes=num_classes, dropout=dropout)
