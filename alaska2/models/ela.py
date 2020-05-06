from pytorch_toolbelt.modules import *
from alaska2.dataset import *

__all__ = ["ela_resnet34"]


class ELAModel(nn.Module):
    def __init__(self, ela_encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        self.ela_bn = nn.BatchNorm2d(15)
        self.ela_encoder = ela_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.embedding = nn.Sequential(
            nn.Linear(self.ela_encoder.channels[-1], 128),
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
        ela = self.ela_bn(kwargs[INPUT_ELA_KEY].float())

        ela_features = self.pool(self.ela_encoder(ela)[-1])
        x = self.embedding(ela_features)

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }


def ela_resnet34(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = Resnet34Encoder(pretrained=pretrained).change_input_channels(15)
    return ELAModel(dct_encoder, num_classes=num_classes, dropout=dropout)
