from collections import OrderedDict

import torch
from pytorch_toolbelt.modules import *
from torch import nn
import torch.nn.functional as F

from alaska2.dataset import *

__all__ = ["dct_seresnext50", "dct_srnet"]


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


class DCTNormalize(Normalize):
    # fmt: on
    def __init__(self):
        super().__init__(
            [
                -1.22e+02, -5.85e+01, 2.65e+01, -1.97e-02, 2.06e-02, -8.85e-03, 1.32e-02, 1.69e-02, 9.04e-03, 2.00e-02,
                7.52e-03,
                3.44e-03, -2.23e-03, 3.08e-03, 1.21e-03, 1.21e-02, 1.57e-03, -5.06e-04, 1.15e-02, 3.51e-03, 2.27e-03,
                1.49e-02,
                -6.46e-04, -1.76e-03, 4.22e-01, 7.45e-02, -5.87e-02, -3.11e-03, -1.49e-03, -1.88e-04, 4.31e-04,
                -1.13e-03, 1.36e-03,
                -1.48e-03, -6.16e-05, -1.16e-03, -1.26e-03, 6.79e-05, -8.04e-05, -2.85e-03, 3.47e-04, -4.87e-04,
                -9.09e-04, 3.23e-04,
                -1.95e-04, -3.51e-03, -9.81e-04, -1.30e-03, 1.71e-03, 3.40e-03, -1.80e-03, -1.48e-04, 1.79e-03,
                6.04e-05, -2.00e-03,
                4.85e-04, -1.06e-03, 8.88e-04, 1.69e-05, 3.27e-04, 2.23e-05, -6.63e-04, -6.50e-04, -6.46e-05,
                -7.52e-05, 5.01e-05,
                1.06e-03, -5.36e-04, -2.58e-04, -1.71e-04, -1.12e-04, -2.80e-04, 3.64e-02, 1.04e-02, -1.09e-02,
                -1.30e-03, -7.97e-04,
                -5.59e-04, 2.57e-05, -4.00e-04, 1.04e-04, -1.68e-03, -1.70e-04, -4.01e-04, -2.04e-03, -3.98e-04,
                5.36e-04, -1.44e-03,
                5.54e-05, -3.59e-04, -6.03e-04, 4.14e-04, 1.30e-05, -4.22e-03, -1.03e-04, -8.11e-04, 3.58e-03,
                6.09e-03, 1.53e-03,
                1.32e-02, 3.52e-03, 2.54e-03, 1.12e-02, 2.55e-03, 2.45e-03, 7.96e-03, 1.41e-03, 1.90e-03, -2.48e-04,
                3.48e-04,
                -2.91e-05, 3.52e-03, 1.37e-03, 1.63e-03, 3.84e-03, 6.66e-04, 7.12e-04, 4.24e-03, 1.41e-03, 1.09e-03,
                -7.87e-03,
                -3.77e-04, -4.10e-03, -1.29e-03, -6.42e-04, -5.20e-04, -1.04e-03, 1.60e-04, -1.60e-05, -6.51e-04,
                -2.64e-04, -3.83e-04,
                8.47e-04, 6.30e-04, 1.60e-04, -1.39e-03, -5.03e-04, -6.71e-04, 5.06e-04, -1.95e-04, -1.43e-04,
                -4.18e-03, -1.22e-03,
                -1.71e-03, 2.80e-03, 2.37e-03, 8.20e-04, -4.94e-04, -3.92e-04, 1.60e-04, 3.05e-04, -1.17e-04, 5.43e-05,
                -4.70e-04,
                -1.17e-04, -6.51e-05, -5.12e-04, 9.03e-05, -3.42e-04, -5.47e-04, 6.65e-05, -1.77e-04, 5.22e-05,
                1.13e-04, 1.57e-04,
                -2.36e-04, 5.12e-04, 5.07e-05, -2.96e-02, 6.22e-05, -5.52e-03, -2.66e-03, -9.94e-04, -9.33e-04,
                -2.32e-05, 1.25e-05,
                -1.40e-04, -3.44e-03, -3.64e-04, -5.13e-04, 1.20e-04, -2.85e-05, -1.12e-04, -3.93e-03, -1.26e-03,
                -2.04e-03, -8.55e-05,
                7.48e-05, 3.93e-05, -9.06e-03, -2.72e-03, -3.38e-03
            ],
            [
                418.89, 132.91, 125.32, 60.86, 17.56, 16.19, 32.42, 10.99, 10.54, 19.84, 7.82, 7.59, 13.96, 5.74, 5.46,
                10.35,
                4.59, 4.28, 8.09, 3.87, 3.56, 6.63, 3.52, 3.19, 64.62, 18.04, 16.51, 32.77, 11.2, 10.56, 21.67, 8.5,
                8.19, 14.81, 6.49, 6.23, 11.12, 5.03, 4.74, 8.68, 4.22, 3.9, 6.95, 3.66, 3.34, 5.83, 3.33, 3.,
                35.39, 11.43, 10.9, 22.42, 8.61, 8.26, 16.66, 7.05, 6.76, 12.49, 5.53, 5.24, 9.91, 4.65, 4.34, 7.95,
                4.04, 3.71, 6.43, 3.57, 3.25, 5.5, 3.26, 2.94, 21.94, 8.25, 7.96, 15.65, 6.67, 6.38, 12.76, 5.62,
                5.31, 10.43, 4.85, 4.54, 8.73, 4.29, 3.97, 7.08, 3.85, 3.52, 5.92, 3.47, 3.15, 5.16, 3.21, 2.89,
                15.45, 6.18, 5.87, 11.84, 5.27, 4.94, 10.27, 4.78, 4.45, 8.86, 4.34, 4.01, 7.65, 3.98, 3.66, 6.23,
                3.67, 3.34, 5.3, 3.37, 3.05, 4.76, 3.15, 2.83, 11.37, 5.06, 4.72, 9.27, 4.49, 4.13, 8.31, 4.21,
                3.86, 7.39, 3.94, 3.6, 6.52, 3.7, 3.37, 5.54, 3.47, 3.14, 4.72, 3.24, 2.92, 4.3, 3.06, 2.75,
                8.92, 4.37, 4., 7.43, 3.95, 3.58, 6.7, 3.76, 3.4, 6.04, 3.58, 3.23, 5.38, 3.42, 3.09, 4.69,
                3.26, 2.93, 4.18, 3.08, 2.77, 3.88, 2.96, 2.64, 7.22, 4., 3.63, 5.99, 3.61, 3.24, 5.48, 3.45,
                3.09, 5.04, 3.32, 2.98, 4.58, 3.2, 2.87, 4.28, 3.09, 2.77, 3.88, 2.97, 2.65, 3.69, 2.94, 2.61
            ]
        )
    # fmt: off


class DCTModel(nn.Module):
    def __init__(self, dct_encoder: EncoderModule, num_classes: int, dropout=0):
        super().__init__()
        self.s2d = SpaceToDepth(block_size=8)
        self.dct_norm = DCTNormalize()
        self.encoder = dct_encoder
        self.pool = GlobalAvgPool2d(flatten=True)
        self.dropout = nn.Dropout(dropout)

        self.type_classifier = nn.Linear(dct_encoder.channels[-1], num_classes)
        self.flag_classifier = nn.Linear(dct_encoder.channels[-1], 1)

    def forward(self, **kwargs):
        dct = kwargs[INPUT_FEATURES_DCT_KEY].float()
        dct = self.s2d(dct)
        dct = self.dct_norm(dct)
        # print(dct.mean(dim=(0, 2, 3)), dct.std(dim=(0, 2, 3)))
        features = self.encoder(dct)
        x = self.pool(features[-1])

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.dropout(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.dropout(x)),
        }

    @property
    def required_features(self):
        return [INPUT_FEATURES_DCT_KEY]



class SRNetEncoder(nn.Module):
    def __init__(self, in_chanels):
        super(SRNetEncoder, self).__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(
            in_channels=in_chanels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Layer 3
        self.layer31 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.layer32 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(16)
        # Layer 4
        self.layer41 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(16)
        self.layer42 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(16)
        # Layer 5
        self.layer51 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(16)
        self.layer52 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(16)
        # Layer 6
        self.layer61 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn61 = nn.BatchNorm2d(16)
        self.layer62 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn62 = nn.BatchNorm2d(16)
        # Layer 7
        self.layer71 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(16)
        self.layer72 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(16)
        # Layer 8
        self.layer81 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn81 = nn.BatchNorm2d(16)
        self.layer82 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(16)
        self.layer83 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn83 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 9
        self.layer91 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.layer92 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(64)
        self.layer93 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn93 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 10
        self.layer101 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn101 = nn.BatchNorm2d(128)
        self.layer102 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn102 = nn.BatchNorm2d(128)
        self.layer103 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn103 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # Layer 11
        self.layer111 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn111 = nn.BatchNorm2d(256)
        self.layer112 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.layer113 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 12
        self.layer121 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn121 = nn.BatchNorm2d(512)
        self.layer122 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn122 = nn.BatchNorm2d(512)

    @property
    def channels(self):
        return [512]

    def forward(self, inputs):
        # Layer 1
        conv = self.layer1(inputs)
        actv = F.relu(self.bn1(conv))
        # Layer 2
        conv = self.layer2(actv)
        actv = F.relu(self.bn2(conv))
        # Layer 3
        conv1 = self.layer31(actv)
        actv1 = F.relu(self.bn31(conv1))
        conv2 = self.layer32(actv1)
        bn = self.bn32(conv2)
        res = torch.add(actv, bn)
        # Layer 4
        conv1 = self.layer41(res)
        actv1 = F.relu(self.bn41(conv1))
        conv2 = self.layer42(actv1)
        bn = self.bn42(conv2)
        res = torch.add(res, bn)
        # Layer 5
        conv1 = self.layer51(res)
        actv1 = F.relu(self.bn51(conv1))
        conv2 = self.layer52(actv1)
        bn = self.bn52(conv2)
        res = torch.add(res, bn)
        # Layer 6
        conv1 = self.layer61(res)
        actv1 = F.relu(self.bn61(conv1))
        conv2 = self.layer62(actv1)
        bn = self.bn62(conv2)
        res = torch.add(res, bn)
        # Layer 7
        conv1 = self.layer71(res)
        actv1 = F.relu(self.bn71(conv1))
        conv2 = self.layer72(actv1)
        bn = self.bn72(conv2)
        res = torch.add(res, bn)
        # Layer 8
        convs = self.layer81(res)
        convs = self.bn81(convs)
        conv1 = self.layer82(res)
        actv1 = F.relu(self.bn82(conv1))
        conv2 = self.layer83(actv1)
        bn = self.bn83(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 9
        convs = self.layer91(res)
        convs = self.bn91(convs)
        conv1 = self.layer92(res)
        actv1 = F.relu(self.bn92(conv1))
        conv2 = self.layer93(actv1)
        bn = self.bn93(conv2)
        pool = self.pool2(bn)
        res = torch.add(convs, pool)
        # Layer 10
        convs = self.layer101(res)
        convs = self.bn101(convs)
        conv1 = self.layer102(res)
        actv1 = F.relu(self.bn102(conv1))
        conv2 = self.layer103(actv1)
        bn = self.bn103(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 11
        convs = self.layer111(res)
        convs = self.bn111(convs)
        conv1 = self.layer112(res)
        actv1 = F.relu(self.bn112(conv1))
        conv2 = self.layer113(actv1)
        bn = self.bn113(conv2)
        pool = self.pool1(bn)
        res = torch.add(convs, pool)
        # Layer 12
        conv1 = self.layer121(res)
        actv1 = F.relu(self.bn121(conv1))
        conv2 = self.layer122(actv1)
        bn = self.bn122(conv2)
        # print("L12:",res.shape)

        return [bn]


def dct_seresnext50(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = SEResNeXt50Encoder(pretrained=pretrained)
    dct_encoder.layer0 = nn.Sequential(
        OrderedDict([("conv1", nn.Conv2d(64 * 3, 64, kernel_size=1)),
                     ("abn1", ABN(64)),
                     ("conv2", nn.Conv2d(64, 64, kernel_size=1)),
                     ("abn2", ABN(64)),
                     ])
    )
    dct_encoder.maxpool = nn.Identity()
    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)


def dct_srnet(num_classes=4, dropout=0, pretrained=True):
    dct_encoder = SRNetEncoder(64)
    dct_encoder.layer1 = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(64 * 3, 64, kernel_size=1))]))

    return DCTModel(dct_encoder, num_classes=num_classes, dropout=dropout)
