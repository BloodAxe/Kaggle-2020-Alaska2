import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt.modules import ABN, EncoderModule, Normalize, GlobalAvgPool2d

__all__ = ["SRNetEncoder", "SRNetModel", "srnet", "srnet_inplace"]

from alaska2.dataset import INPUT_IMAGE_KEY, OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_MODIFICATION_TYPE
from .sa import SelfAttention


class Layer1(nn.Module):
    def __init__(self, in_channels, out_channels, abn=ABN):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn = abn(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class Layer2(nn.Module):
    def __init__(self, channels, abn=ABN):
        super().__init__()
        self.conv1 = Layer1(channels, channels, abn)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + skip


class Layer3(nn.Module):
    def __init__(self, in_channels, out_channels, abn=ABN):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.main = nn.Sequential(
            Layer1(in_channels, out_channels, abn),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.residual(x) + self.main(x)


class SRNetEncoder(EncoderModule):
    def __init__(self, in_chanels, abn=ABN):
        super(SRNetEncoder, self).__init__([512], [32], [0])
        # Layer 1
        self.layer1 = nn.Sequential(Layer1(in_chanels, 64, abn), Layer1(64, 16, abn))
        self.layer2 = nn.Sequential(
            Layer2(16, abn), Layer2(16, abn), Layer2(16, abn), Layer2(16, abn), Layer2(16, abn)
        )
        self.layer3 = nn.Sequential(
            Layer3(16, 64, abn), Layer3(64, 128, abn), Layer3(128, 256, abn), Layer3(256, 512, abn)
        )
        self.layer4 = nn.Sequential(
            Layer1(512, 512), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512)
        )

    @property
    def channels(self):
        return [512]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return [x]


class SRNetModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_classes, dropout=0):
        super().__init__()
        self.encoder = encoder
        max_pixel_value = 255
        self.rgb_bn = Normalize(max_pixel_value, 1)
        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.channels[-1], num_classes)
        self.flag_classifier = nn.Linear(encoder.channels[-1], 1)

    def forward(self, **kwargs):
        x = kwargs[INPUT_IMAGE_KEY]
        x = self.rgb_bn(x.float())
        x = self.encoder(x)
        x = self.pool(x[-1])
        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY]


def srnet(num_classes=4, pretrained=True, dropout=0):
    return SRNetModel(encoder=SRNetEncoder(3), num_classes=num_classes, dropout=dropout)


def srnet_inplace(num_classes=4, pretrained=True, dropout=0):
    import inplace_abn

    return SRNetModel(encoder=SRNetEncoder(3, abn=inplace_abn.InPlaceABN), num_classes=num_classes, dropout=dropout)
