import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn

from inplace_abn import ABN
from pytorch_toolbelt.modules import GlobalAvgPool2d

from .residual import IdentityResidualBlock


# "16": {"structure": [1, 1, 1, 1, 1, 1]},
#     "20": {"structure": [1, 1, 1, 3, 1, 1]},
#     "38": {"structure": },

__all__ = ["wider_resnet38", "wider_resnet38_a2"]


class WiderResNet(nn.Module):
    def __init__(self, structure, in_chans=3, norm_act=ABN, classes=0):
        """Wider ResNet with pre-activation (identity mapping) blocks
        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        # self.mod1 = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))

        # Deep stem
        self.mod1 = nn.Sequential(
            *[
                nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False),
                norm_act(64),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                norm_act(64),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            ]
        )

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act),
                    )
                )

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id <= 4:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(
                OrderedDict([("avg_pool", GlobalAvgPool2d(flatten=True)), ("fc", nn.Linear(in_channels, classes))])
            )
        self.num_features = in_channels

    def forward_features(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)
        return out

    def forward(self, img):
        out = self.forward_features(img)
        if hasattr(self, "classifier"):
            out = self.classifier(out)

        return out


class WiderResNetA2(nn.Module):
    def __init__(self, structure, in_chans=3, norm_act=ABN, classes=0, dilation=False):
        """Wider ResNet with pre-activation (identity mapping) blocks
        This variant uses down-sampling by max-pooling in the first two blocks and by strided convolution in the others.
        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the down-sampling factor from 32 to 8.
        """
        super(WiderResNetA2, self).__init__()
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.5)
                else:
                    drop = None

                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        IdentityResidualBlock(
                            in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil, dropout=drop
                        ),
                    )
                )

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(
                OrderedDict([("avg_pool", GlobalAvgPool2d(flatten=True)), ("fc", nn.Linear(in_channels, classes))])
            )

        self.num_features = in_channels

    def forward_features(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)
        return out

    def forward(self, img):
        out = self.forward_features(img)

        if hasattr(self, "classifier"):
            return self.classifier(out)
        else:
            return out


def wider_resnet38(in_chans: int):
    return WiderResNet(structure=[3, 3, 6, 3, 1, 1], in_chans=in_chans)


def wider_resnet38_a2(in_chans: int):
    return WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], in_chans=in_chans)
