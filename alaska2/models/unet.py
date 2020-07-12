from typing import Union, List

import numpy as np
import torch
from pytorch_toolbelt.modules import EncoderModule, DecoderModule, make_n_channel_input, Normalize, GlobalAvgPool2d
from timm.models.efficientnet_blocks import InvertedResidual, DepthwiseSeparableConv
from timm.models.layers import Mish
from torch import nn

from alaska2.dataset import (
    INPUT_IMAGE_KEY,
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_MASK,
    OUTPUT_PRED_MODIFICATION_TYPE,
    OUTPUT_PRED_PAYLOAD_BITS,
    INPUT_FEATURES_JPEG_FLOAT,
)

__all__ = ["nr_rgb_unet"]


class EfficientUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_layer=Mish, stride=1):
        super().__init__()
        self.ds1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride, act_layer=act_layer)
        self.ir1 = InvertedResidual(out_channels, out_channels, act_layer=act_layer, se_ratio=0.25, drop_path_rate=0.1)
        self.ds2 = DepthwiseSeparableConv(out_channels, out_channels, act_layer=act_layer)
        self.ir2 = InvertedResidual(out_channels, out_channels, act_layer=act_layer, se_ratio=0.25, drop_path_rate=0.1)

    def forward(self, x):
        x = self.ds1(x)
        x = self.ir1(x)
        x = self.ds2(x)
        x = self.ir2(x)
        return x


class EfficientUnetEncoder(EncoderModule):
    """
    Efficient-net style U-Net encoder
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        num_layers=4,
        growth_factor=2,
        unet_block: Union[nn.Module, EfficientUnetBlock] = EfficientUnetBlock,
    ):
        feature_maps = [out_channels * (growth_factor ** i) for i in range(num_layers)]
        strides = [2 ** i for i in range(num_layers)]
        super().__init__(feature_maps, strides, layers=list(range(num_layers)))

        input_filters = in_channels
        self.num_layers = num_layers
        for layer in range(num_layers):
            block = unet_block(input_filters, feature_maps[layer], stride=2 if layer > 0 else 1)
            input_filters = feature_maps[layer]
            self.add_module(f"layer{layer}", block)

    @property
    def encoder_layers(self):
        return [self.__getattr__(f"layer{layer}") for layer in range(self.num_layers)]

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


class EfficientUNetDecoder(DecoderModule):
    def __init__(
        self,
        feature_maps: List[int],
        decoder_features: Union[int, List[int]] = None,
        unet_block=EfficientUnetBlock,
        upsample_block: Union[nn.Upsample, nn.ConvTranspose2d] = None,
    ):
        super().__init__()

        # if not isinstance(decoder_features, list):
        #     decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]
        # else:
        #     assert len(decoder_features) == len(
        #         feature_maps
        #     ), f"Incorrect number of decoder features: {decoder_features}, {feature_maps}"

        if upsample_block is None:
            upsample_block = nn.UpsamplingBilinear2d

        blocks = []
        upsamples = []

        num_blocks = len(feature_maps) - 1  # Number of outputs is one less than encoder layers

        if decoder_features is None:
            decoder_features = [None] * num_blocks
        else:
            if len(decoder_features) != num_blocks:
                raise ValueError(f"decoder_features must have length of {num_blocks}")
        in_channels_for_upsample_block = feature_maps[-1]

        for block_index in reversed(range(num_blocks)):
            features_from_encoder = feature_maps[block_index]

            if isinstance(upsample_block, nn.Upsample):
                upsamples.append(upsample_block)
                out_channels_from_upsample_block = in_channels_for_upsample_block
            elif issubclass(upsample_block, nn.Upsample):
                upsamples.append(upsample_block(scale_factor=2))
                out_channels_from_upsample_block = in_channels_for_upsample_block
            elif issubclass(upsample_block, nn.ConvTranspose2d):
                up = upsample_block(
                    in_channels_for_upsample_block,
                    in_channels_for_upsample_block // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                upsamples.append(up)
                out_channels_from_upsample_block = up.out_channels
            else:
                up = upsample_block(in_channels_for_upsample_block)
                upsamples.append(up)
                out_channels_from_upsample_block = up.out_channels

            in_channels = features_from_encoder + out_channels_from_upsample_block
            out_channels = decoder_features[block_index] or in_channels // 2
            blocks.append(unet_block(in_channels, out_channels))

            in_channels_for_upsample_block = out_channels
            decoder_features[block_index] = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.upsamples = nn.ModuleList(upsamples)
        self.output_filters = decoder_features

    @property
    def channels(self) -> List[int]:
        return self.output_filters

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        x = feature_maps[-1]
        outputs = []
        num_feature_maps = len(feature_maps)
        for index, (upsample_block, decoder_block) in enumerate(zip(self.upsamples, self.blocks)):
            encoder_input = feature_maps[num_feature_maps - index - 2]

            if isinstance(upsample_block, nn.ConvTranspose2d):
                x = upsample_block(x, output_size=encoder_input.size())
            else:
                x = upsample_block(x)

            x = torch.cat([x, encoder_input], dim=1)
            x = decoder_block(x)
            outputs.append(x)

        # Returns list of tensors in same order as input (fine-to-coarse)
        return outputs[::-1]


class UnetModel(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
        max_pixel_value=255,
        input_key=INPUT_IMAGE_KEY,
    ):
        super().__init__()
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)

        self.encoder = EfficientUnetEncoder()
        self.decoder = EfficientUNetDecoder(self.encoder.channels, decoder_features=[16, 32, 64])

        self.pool = GlobalAvgPool2d(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(self.encoder.channels[-1], num_classes)
        self.flag_classifier = nn.Linear(self.encoder.channels[-1], 1)
        self.input_key = input_key

        self.mask = nn.Conv2d(self.decoder.channels[0], 1, kernel_size=1)

    def forward(self, **kwargs):
        x = kwargs[self.input_key]
        x = self.rgb_bn(x)

        features = self.encoder(x)
        decoded = self.decoder(features)

        # Pool last feature map for classifier
        pooled = self.pool(features[-1])

        mask = self.mask(decoded[0])

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(pooled)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(pooled)),
            OUTPUT_PRED_MODIFICATION_MASK: mask,
            OUTPUT_PRED_PAYLOAD_BITS: mask.sigmoid().sum(dim=(2, 3)),
        }

    @property
    def required_features(self):
        return [self.input_key]


def nr_rgb_unet(num_classes=4, pretrained=True, dropout=0.2):
    return UnetModel(num_classes=num_classes, dropout=dropout, input_key=INPUT_FEATURES_JPEG_FLOAT)
