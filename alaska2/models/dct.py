from collections import OrderedDict
from pytorch_toolbelt.modules import *
from torch import nn

from alaska2.dataset import *

__all__ = ["dct_seresnext50"]


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class DCTModel(nn.Module):
    def __init__(self, dct_encoder: EncoderModule, num_classes: int, dropout=0):
        super().__init__()
        self.encoder = dct_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.dropout = nn.Dropout(dropout)

        self.type_classifier = nn.Linear(dct_encoder.channels[-1], num_classes)
        self.flag_classifier = nn.Linear(dct_encoder.channels[-1], 1)

    def forward(self, **kwargs):
        dct = kwargs[INPUT_FEATURES_DCT_KEY].float()
        features = self.encoder(dct)
        x = self.pool(features[-1])

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.dropout(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.dropout(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_DCT_KEY]


def dct_seresnext50(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = SEResNeXt50Encoder(pretrained=pretrained)
    dct_encoder.layer0 = nn.Sequential(
        OrderedDict(
            [("s2d", SpaceToDepth(block_size=8)), ("conv1", nn.Conv2d(64 * 3, 64, kernel_size=1)), ("abn1", ABN(64))]
        )
    )

    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)
