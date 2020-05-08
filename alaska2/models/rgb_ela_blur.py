from pytorch_toolbelt.modules import *
from pytorch_toolbelt.modules import encoders as E
from alaska2.dataset import *

__all__ = ["frank"]


import math
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import List

import torch
from torch import nn


def round_filters(filters: int, width_coefficient, depth_divisor, min_depth) -> int:
    """
    Calculate and round number of filters based on depth multiplier.
    """
    filters *= width_coefficient
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_multiplier):
    """
    Round number of filters based on depth multiplier.
    """
    if not depth_multiplier:
        return repeats
    return int(math.ceil(depth_multiplier * repeats))


def drop_connect(inputs, p, training):
    """
    Drop connect implementation.
    """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class EfficientNetBlockArgs:
    def __init__(
        self,
        input_filters,
        output_filters,
        expand_ratio,
        repeats=1,
        kernel_size=3,
        stride=1,
        se_reduction=4,
        dropout=0.0,
        id_skip=True,
    ):
        self.in_channels = input_filters
        self.out_channels = output_filters
        self.expand_ratio = expand_ratio
        self.num_repeat = repeats
        self.se_reduction = se_reduction
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride
        self.width_coefficient = 1.0
        self.depth_coefficient = 1.0
        self.depth_divisor = 8
        self.min_filters = None
        self.id_skip = id_skip

    def __repr__(self):
        """Encode a block args class to a string representation."""
        args = [
            "r%d" % self.num_repeat,
            "k%d" % self.kernel_size,
            "s%d" % self.stride,
            "e%s" % self.expand_ratio,
            "i%d" % self.in_channels,
            "o%d" % self.out_channels,
        ]
        if self.se_reduction > 0:
            args.append("se%s" % self.se_reduction)
        return "_".join(args)

    def copy(self):
        return deepcopy(self)

    def scale(
        self, width_coefficient: float, depth_coefficient: float, depth_divisor: float = 8.0, min_filters: int = None
    ):
        copy = self.copy()
        copy.in_channels = round_filters(self.in_channels, width_coefficient, depth_divisor, min_filters)
        copy.out_channels = round_filters(self.out_channels, width_coefficient, depth_divisor, min_filters)
        copy.num_repeat = round_repeats(self.num_repeat, depth_coefficient)
        copy.width_coefficient = width_coefficient
        copy.depth_coefficient = depth_coefficient
        copy.depth_divisor = depth_divisor
        copy.min_filters = min_filters
        return copy

    @staticmethod
    def B0():
        params = get_default_efficientnet_params(dropout=0.2)
        params = [p.scale(width_coefficient=1.0, depth_coefficient=1.0) for p in params]
        return params

    @staticmethod
    def B1():
        params = get_default_efficientnet_params(dropout=0.2)
        params = [p.scale(width_coefficient=1.0, depth_coefficient=1.1) for p in params]
        return params

    @staticmethod
    def B2():
        params = get_default_efficientnet_params(dropout=0.3)
        params = [p.scale(width_coefficient=1.1, depth_coefficient=1.2) for p in params]
        return params

    @staticmethod
    def B3():
        params = get_default_efficientnet_params(dropout=0.3)
        params = [p.scale(width_coefficient=1.2, depth_coefficient=1.4) for p in params]
        return params

    @staticmethod
    def B4():
        params = get_default_efficientnet_params(dropout=0.4)
        params = [p.scale(width_coefficient=1.4, depth_coefficient=1.8) for p in params]
        return params

    @staticmethod
    def B5():
        params = get_default_efficientnet_params(dropout=0.4)
        params = [p.scale(width_coefficient=1.6, depth_coefficient=2.2) for p in params]
        return params

    @staticmethod
    def B6():
        params = get_default_efficientnet_params(dropout=0.5)
        params = [p.scale(width_coefficient=1.8, depth_coefficient=2.6) for p in params]
        return params

    @staticmethod
    def B7():
        params = get_default_efficientnet_params(dropout=0.5)
        params = [p.scale(width_coefficient=2.0, depth_coefficient=3.1) for p in params]
        return params


def get_default_efficientnet_params(dropout=0.2) -> List[EfficientNetBlockArgs]:
    #  _DEFAULT_BLOCKS_ARGS = [
    #     'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    #     'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    #     'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    #     'r1_k3_s11_e6_i192_o320_se0.25',
    # ]
    return [
        EfficientNetBlockArgs(
            repeats=1, kernel_size=3, stride=1, expand_ratio=1, input_filters=32, output_filters=16, dropout=dropout
        ),
        EfficientNetBlockArgs(
            repeats=2, kernel_size=3, stride=2, expand_ratio=6, input_filters=16, output_filters=24, dropout=dropout
        ),
        EfficientNetBlockArgs(
            repeats=2, kernel_size=5, stride=2, expand_ratio=6, input_filters=24, output_filters=40, dropout=dropout
        ),
        EfficientNetBlockArgs(
            repeats=3, kernel_size=3, stride=2, expand_ratio=6, input_filters=40, output_filters=80, dropout=dropout
        ),
        EfficientNetBlockArgs(
            repeats=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=80, output_filters=112, dropout=dropout
        ),
        EfficientNetBlockArgs(
            repeats=4, kernel_size=5, stride=2, expand_ratio=6, input_filters=112, output_filters=192, dropout=dropout
        ),
        EfficientNetBlockArgs(
            repeats=1, kernel_size=3, stride=1, expand_ratio=6, input_filters=192, output_filters=320, dropout=dropout
        ),
    ]


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args: EfficientNetBlockArgs, abn_block: ABN):
        super().__init__()

        self.has_se = block_args.se_reduction is not None
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self.expand_ratio = block_args.expand_ratio
        self.stride = block_args.stride

        # Expansion phase
        inp = block_args.in_channels  # number of input channels
        oup = block_args.in_channels * block_args.expand_ratio  # number of output channels

        if block_args.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.abn0 = abn_block(oup)

        # Depthwise convolution phase
        self.depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=block_args.kernel_size,
            padding=block_args.kernel_size // 2,
            stride=block_args.stride,
            bias=False,
        )
        self.abn1 = abn_block(oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            se_channels = max(1, inp // block_args.se_reduction)
            self.se_block = SpatialGate2d(oup, squeeze_channels=se_channels)

        # Output phase
        self.project_conv = nn.Conv2d(in_channels=oup, out_channels=block_args.out_channels, kernel_size=1, bias=False)
        self.abn2 = abn_block(block_args.out_channels)

        self.input_filters = block_args.in_channels
        self.output_filters = block_args.out_channels

        self.reset_parameters()

    def reset_parameters(self):
        pass

    #     if hasattr(self, "expand_conv"):
    #         torch.nn.init.kaiming_uniform_(
    #             self.expand_conv.weight,
    #             a=abn_params.get("slope", 0),
    #             nonlinearity=sanitize_activation_name(self.abn2["activation"]),
    #         )
    #
    #     torch.nn.init.kaiming_uniform_(
    #         self.depthwise_conv.weight,
    #         a=abn_params.get("slope", 0),
    #         nonlinearity=sanitize_activation_name(abn_params["activation"]),
    #     )
    #     torch.nn.init.kaiming_uniform_(
    #         self.project_conv.weight,
    #         a=abn_params.get("slope", 0),
    #         nonlinearity=sanitize_activation_name(abn_params["activation"]),
    #     )

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self.expand_ratio != 1:
            # Expansion and Depthwise Convolution
            x = self.abn0(self.expand_conv(inputs))

        x = self.abn1(self.depthwise_conv(x))

        # Squeeze and Excitation
        if self.has_se:
            x = self.se_block(x)

        x = self.abn2(self.project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNetStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block: ABN):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.abn = abn_block(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class FrankEncoder(nn.Module):
    @staticmethod
    def build_layer(block_args: List[EfficientNetBlockArgs], abn_block: ABN):
        blocks = []
        for block_index, cfg in enumerate(block_args):
            module = []
            # The first block needs to take care of stride and filter size increase.
            module.append(("mbconv_0", MBConvBlock(cfg, abn_block)))

            if cfg.num_repeat > 1:
                cfg = cfg.copy()
                cfg.stride = 1
                cfg.in_channels = cfg.out_channels

                for i in range(cfg.num_repeat - 1):
                    module.append((f"mbconv_{i + 1}", MBConvBlock(cfg, abn_block)))

            module = nn.Sequential(OrderedDict(module))
            blocks.append((f"block_{block_index}", module))

        return nn.Sequential(OrderedDict(blocks))

    def __init__(
        self, rgb_channels=3, ela_features=15, dct_features=64, blur_features=9, dropout=0.0, activation=ACT_SWISH
    ):
        super().__init__()

        abn_block = partial(ABN, activation=activation)
        dropout = dropout

        self.rgb_stem = nn.Sequential(
            EfficientNetStem(rgb_channels, 16, abn_block),
            self.build_layer(
                [
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=2,
                        expand_ratio=1,
                        input_filters=16,
                        output_filters=32,
                        dropout=dropout,
                    ),
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=2,
                        expand_ratio=1,
                        input_filters=32,
                        output_filters=64,
                        dropout=dropout,
                    ),
                ],
                abn_block,
            ),
        )

        self.ela_stem = nn.Sequential(
            EfficientNetStem(ela_features, 16, abn_block),
            self.build_layer(
                [
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=2,
                        expand_ratio=1,
                        input_filters=16,
                        output_filters=32,
                        dropout=dropout,
                    ),
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=2,
                        expand_ratio=1,
                        input_filters=32,
                        output_filters=64,
                        dropout=dropout,
                    ),
                ],
                abn_block,
            ),
        )

        self.blur_stem = nn.Sequential(
            EfficientNetStem(blur_features, 16, abn_block),
            self.build_layer(
                [
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=2,
                        expand_ratio=1,
                        input_filters=16,
                        output_filters=32,
                        dropout=dropout,
                    ),
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=2,
                        expand_ratio=1,
                        input_filters=32,
                        output_filters=64,
                        dropout=dropout,
                    ),
                ],
                abn_block,
            ),
        )

        self.dct_stem = nn.Sequential(
            self.build_layer(
                [
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=1,
                        expand_ratio=1,
                        input_filters=dct_features,
                        output_filters=32,
                        dropout=dropout,
                    ),
                    EfficientNetBlockArgs(
                        repeats=2,
                        kernel_size=3,
                        stride=1,
                        expand_ratio=1,
                        input_filters=32,
                        output_filters=64,
                        dropout=dropout,
                    ),
                ],
                abn_block,
            )
        )

        blocks2layers = [
            EfficientNetBlockArgs(
                repeats=2,
                kernel_size=3,
                stride=2,
                expand_ratio=1,
                input_filters=256,
                output_filters=256,
                dropout=dropout,
            ),
            EfficientNetBlockArgs(
                repeats=3,
                kernel_size=3,
                stride=2,
                expand_ratio=1,
                input_filters=256,
                output_filters=256,
                dropout=dropout,
            ),
            EfficientNetBlockArgs(
                repeats=4,
                kernel_size=3,
                stride=2,
                expand_ratio=1,
                input_filters=256,
                output_filters=256,
                dropout=dropout,
            ),
        ]
        self.encoder = self.build_layer(blocks2layers, abn_block)

    def forward(self, rgb, blur_features, dct_features, ela_features) -> torch.Tensor:  # skipcq: PYL-W0221
        rgb_features = self.rgb_stem(rgb)
        blur_features = self.blur_stem(blur_features)
        ela_features = self.ela_stem(ela_features)
        dct_features = self.dct_stem(dct_features)

        x = torch.cat([rgb_features, blur_features, ela_features, dct_features], dim=1)
        x = self.encoder(x)
        return x


class FrankModel(nn.Module):
    def __init__(self, num_classes, dropout=0.2, activation=ACT_SWISH):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(
            [0.3914976 * max_pixel_value, 0.44266784 * max_pixel_value, 0.46043398 * max_pixel_value],
            [0.17819773 * max_pixel_value, 0.17319807 * max_pixel_value, 0.18128773 * max_pixel_value],
        )

        self.encoder = FrankEncoder(activation=activation, dropout=dropout)
        self.pool = GlobalAvgPool2d(flatten=True)
        activation = get_activation_block(activation)
        self.embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.AlphaDropout(dropout),
            activation(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            activation(),
        )

        self.type_classifier = nn.Linear(128, num_classes)
        self.flag_classifier = nn.Linear(128, 1)

    def forward(self, **kwargs):
        rgb = kwargs[INPUT_IMAGE_KEY]
        ela = kwargs[INPUT_FEATURES_ELA_KEY]
        dct = kwargs[INPUT_FEATURES_DCT_KEY]
        blur = kwargs[INPUT_FEATURES_BLUR_KEY]

        features = self.encoder(self.rgb_bn(rgb.float()), blur, dct, ela)
        embedding = self.pool(features)
        x = self.embedding(embedding)

        return {
            OUTPUT_PRED_EMBEDDING: embedding,
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(x),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(x),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY, INPUT_FEATURES_ELA_KEY, INPUT_FEATURES_BLUR_KEY, INPUT_FEATURES_DCT_KEY]


def frank(num_classes=4, dropout=0, pretrained=True):
    return FrankModel(num_classes=num_classes, dropout=dropout)
