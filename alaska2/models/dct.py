from pytorch_toolbelt.modules import *
from alaska2.dataset import *

__all__ = ["dct_resnet34"]


class DCTModel(nn.Module):
    def __init__(self, dct_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        self.dct_bn = nn.BatchNorm2d(64)
        self.dct_encoder = dct_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.embedding = nn.Sequential(
            nn.Linear(self.dct_encoder.channels[-1], 128),
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
        dct = self.dct_bn(kwargs[INPUT_DCT_KEY].float())

        dct_featues = self.pool(self.dct_encoder(dct)[-1])
        x = self.embedding(dct_featues)

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }


def dct_resnet34(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = Resnet34Encoder(pretrained=pretrained).change_input_channels(64)
    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)
