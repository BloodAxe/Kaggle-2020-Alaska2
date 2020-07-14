from collections import OrderedDict

import numpy as np
import torch
from pytorch_toolbelt.modules import Normalize, GlobalAvgPool2d
from timm.models import skresnext50_32x4d
from timm.models import tresnet, efficientnet, resnet
from timm.models.layers import Swish, Mish

from torch import nn
from torch.nn import InstanceNorm2d

from alaska2.loss import ArcMarginProduct
from alaska2.dataset import (
    INPUT_IMAGE_KEY,
    OUTPUT_PRED_MODIFICATION_FLAG,
    OUTPUT_PRED_MODIFICATION_TYPE,
    INPUT_FEATURES_JPEG_FLOAT,
    OUTPUT_PRED_EMBEDDING, OUTPUT_PRED_EMBEDDING_ARC_MARGIN,
)

__all__ = [
    "rgb_skresnext50_32x4d",
    "rgb_tf_efficientnet_b6_ns",
    "rgb_tf_efficientnet_b1_ns",
    "rgb_swsl_resnext101_32x8d",
    "rgb_tf_efficientnet_b2_ns",
    "rgb_tf_efficientnet_b3_ns",
    "rgb_tresnet_m_448",
    "rgb_tf_efficientnet_b7_ns",
    # Models using unrounded image
    "nr_rgb_tf_efficientnet_b3_ns_mish",
    "nr_rgb_tf_efficientnet_b6_ns",
    "nr_rgb_tf_efficientnet_b6_ns_mish",
    "nr_rgb_tf_efficientnet_b6_ns_mish_gep",
    "nr_rgb_tf_efficientnet_b7_ns_mish",
    "nr_rgb_mixnet_xl",
    "nr_rgb_mixnet_xxl",
    "nr_rgb_tf_efficientnet_b3_ns_gn_mish",
    "nr_rgb_tf_efficientnet_b3_ns_in_mish",
]

from torch.nn.parameter import Parameter
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeneralizedMeanPooling2d(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeneralizedMeanPooling2d, self).__init__()
        self.p = Parameter(torch.ones(1) * p, requires_grad=True)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class TimmRgbModel(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        dropout=0.0,
        mean=[0.3914976, 0.44266784, 0.46043398],
        std=[0.17819773, 0.17319807, 0.18128773],
        max_pixel_value=255,
        input_key=INPUT_IMAGE_KEY,
        need_embedding=False,
        pooling_module=GlobalAvgPool2d,
        arc_margin=None
    ):
        super().__init__()
        self.encoder = encoder

        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)
        self.pool = pooling_module(flatten=True)
        self.drop = nn.Dropout(dropout)
        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)
        self.input_key = input_key
        self.need_embedding = need_embedding
        self.arc_margin = arc_margin

    def forward(self, **kwargs):
        x = kwargs[self.input_key]
        x = self.rgb_bn(x)
        x = self.encoder.forward_features(x)
        embedding = self.pool(x)
        result = {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(embedding)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(embedding)),
        }
        if self.need_embedding:
            result[OUTPUT_PRED_EMBEDDING] = embedding
        if self.arc_margin is not None:
            result[OUTPUT_PRED_EMBEDDING_ARC_MARGIN] = self.arc_margin(embedding)

        return result

    @property
    def required_features(self):
        return [self.input_key]


## Cherry-picked & modified functions from TIMM/EfficientNet to use Mish


def patched_gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"],
    ]
    from timm.models.efficientnet import _create_model, default_cfgs
    from timm.models.efficientnet_builder import decode_arch_def
    from timm.models.efficientnet_blocks import round_channels, resolve_bn_args

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        variant=variant,
        # **kwargs,
    )
    # Update of model_kwargs allows to override activation
    model_kwargs.update(kwargs)
    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def patched_tf_efficientnet_b3_ns(pretrained=False, **kwargs):
    """ EfficientNet-B3. Tensorflow compatible variant """
    from timm.models.efficientnet_blocks import BN_EPS_TF_DEFAULT

    kwargs["bn_eps"] = BN_EPS_TF_DEFAULT
    kwargs["pad_type"] = "same"
    model = patched_gen_efficientnet(
        "tf_efficientnet_b3_ns", channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs
    )
    return model


def patched_tf_efficientnet_b6_ns(pretrained=False, **kwargs):
    """ EfficientNet-B6 NoisyStudent. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    from timm.models.efficientnet_blocks import BN_EPS_TF_DEFAULT

    kwargs["bn_eps"] = BN_EPS_TF_DEFAULT
    kwargs["pad_type"] = "same"
    model = patched_gen_efficientnet(
        "tf_efficientnet_b6_ns", channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs
    )
    return model


def patched_tf_efficientnet_b7_ns(pretrained=False, **kwargs):
    """ EfficientNet-B6 NoisyStudent. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    from timm.models.efficientnet_blocks import BN_EPS_TF_DEFAULT

    kwargs["bn_eps"] = BN_EPS_TF_DEFAULT
    kwargs["pad_type"] = "same"
    model = patched_gen_efficientnet(
        "tf_efficientnet_b7_ns", channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs
    )
    return model


def patched_gen_mixnet_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Medium-Large model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ["ds_r1_k3_s1_e1_c24"],  # relu
        # stage 1, 112x112 in
        ["ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c32", "ir_r1_k3_a1.1_p1.1_s1_e3_c32"],  # relu
        # stage 2, 56x56 in
        ["ir_r1_k3.5.7.9_s2_e6_c40_se0.5_nsw", "ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw"],  # swish
        # stage 3, 28x28 in
        ["ir_r1_k3.5.7_s2_e6_c80_se0.25_nsw", "ir_r3_k3.5.7.9_a1.1_p1.1_s1_e6_c80_se0.25_nsw"],  # swish
        # stage 4, 14x14in
        ["ir_r1_k3_s1_e6_c120_se0.5_nsw", "ir_r3_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw"],  # swish
        # stage 5, 14x14in
        ["ir_r1_k3.5.7.9_s2_e6_c200_se0.5_nsw", "ir_r3_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw"],  # swish
        # 7x7
    ]

    from timm.models.efficientnet import _create_model, default_cfgs
    from timm.models.efficientnet_builder import decode_arch_def
    from timm.models.efficientnet_blocks import round_channels, resolve_bn_args

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc="round"),
        num_features=1536,
        stem_size=24,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        # **kwargs
    )
    # Update of model_kwargs allows to override activation
    model_kwargs.update(kwargs)

    model = _create_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


def patched_mixnet_xxl(pretrained=False, **kwargs):
    """Creates a MixNet Double Extra Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    model = patched_gen_mixnet_m(
        "mixnet_xxl", channel_multiplier=2.4, depth_multiplier=1.3, pretrained=pretrained, **kwargs
    )
    return model


# Model zoo


def rgb_skresnext50_32x4d(num_classes=4, pretrained=True, dropout=0):
    encoder = skresnext50_32x4d(pretrained=pretrained)
    del encoder.fc

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_tresnet_m_448(num_classes=4, pretrained=True, dropout=0):
    encoder = tresnet.tresnet_m_448(pretrained=pretrained)
    del encoder.head

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_swsl_resnext101_32x8d(num_classes=4, pretrained=True, dropout=0):
    encoder = resnet.swsl_resnext101_32x8d(pretrained=pretrained)
    del encoder.fc

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_tf_efficientnet_b1_ns(num_classes=4, pretrained=True, dropout=0.1):
    encoder = efficientnet.tf_efficientnet_b1_ns(pretrained=pretrained, drop_path_rate=0.1)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def rgb_tf_efficientnet_b2_ns(num_classes=4, pretrained=True, dropout=0.1):
    encoder = efficientnet.tf_efficientnet_b2_ns(pretrained=pretrained, drop_path_rate=0.1)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_tf_efficientnet_b3_ns(num_classes=4, pretrained=True, dropout=0.1):
    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=pretrained, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_tf_efficientnet_b6_ns(num_classes=4, pretrained=True, dropout=0.5):
    encoder = efficientnet.tf_efficientnet_b6_ns(pretrained=pretrained, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(encoder, num_classes=num_classes, dropout=dropout)


def rgb_tf_efficientnet_b7_ns(num_classes=4, pretrained=True, dropout=0.5):
    encoder = efficientnet.tf_efficientnet_b7_ns(pretrained=pretrained, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def nr_rgb_tf_efficientnet_b3_ns_mish(num_classes=4, pretrained=True, dropout=0.2, need_embedding=False):
    encoder = patched_tf_efficientnet_b3_ns(pretrained=pretrained, act_layer=Mish, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )


def nr_rgb_tf_efficientnet_b3_ns_gn_mish(num_classes=4, pretrained=True, dropout=0.2, need_embedding=False):
    def group_norm_builder(channels, **kwargs):
        groups = [32, 24, 16, 8, 7, 6, 5, 4, 3, 2]
        for g in groups:
            if channels % g == 0 and channels > g:
                norm_layer = nn.GroupNorm(g, channels)
                norm_layer.weight.data.fill_(1.0)
                norm_layer.bias.data.zero_()
                print(f"Created nn.GroupNorm(groups={g}, channels={channels})")
                # Return sequential with gn to prevent copying weights from BN
                return nn.Sequential(OrderedDict([("gn", norm_layer)]))
        raise ValueError(f"Cannot create GroupNorm layer with number of channels {channels}")

    encoder = patched_tf_efficientnet_b3_ns(
        pretrained=pretrained, drop_path_rate=0.2, norm_layer=group_norm_builder, act_layer=Mish
    )
    print(encoder)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )


def nr_rgb_tf_efficientnet_b3_ns_in_mish(num_classes=4, pretrained=True, dropout=0.2, need_embedding=False):
    def instance_norm_builder(channels, **kwargs):
        norm_layer = InstanceNorm2d(channels, affine=True)
        return nn.Sequential(OrderedDict([("in", norm_layer)]))

    encoder = patched_tf_efficientnet_b3_ns(
        pretrained=pretrained, drop_path_rate=0.2, norm_layer=instance_norm_builder, act_layer=Mish
    )
    print(encoder)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )


def nr_rgb_tf_efficientnet_b6_ns(num_classes=4, pretrained=True, dropout=0.5, need_embedding=False):
    encoder = efficientnet.tf_efficientnet_b6_ns(pretrained=pretrained, drop_path_rate=0.2)
    del encoder.classifier
    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )


def nr_rgb_tf_efficientnet_b6_ns_mish(num_classes=4, pretrained=True, dropout=0.5, need_embedding=False):
    encoder = patched_tf_efficientnet_b6_ns(pretrained=pretrained, act_layer=Mish, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )


def nr_rgb_tf_efficientnet_b6_ns_mish_gep(num_classes=4, pretrained=True, dropout=0.5, need_embedding=False):
    encoder = patched_tf_efficientnet_b6_ns(pretrained=pretrained, act_layer=Mish, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
        pooling_module=GeneralizedMeanPooling2d,
        arc_margin=ArcMarginProduct(encoder.num_features, num_classes) if need_embedding else None
    )


def nr_rgb_tf_efficientnet_b7_ns_mish(num_classes=4, pretrained=True, dropout=0.5, need_embedding=False):
    encoder = patched_tf_efficientnet_b7_ns(pretrained=pretrained, act_layer=Mish, drop_path_rate=0.2)
    del encoder.classifier

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )


def nr_rgb_mixnet_xl(num_classes=4, pretrained=True, dropout=0.5, need_embedding=False):
    encoder = efficientnet.mixnet_xl(pretrained=pretrained, drop_path_rate=0.2)
    del encoder.classifier
    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )


def nr_rgb_mixnet_xxl(num_classes=4, pretrained=True, dropout=0.5, need_embedding=False):
    encoder = patched_mixnet_xxl(pretrained=pretrained, act_layer=Mish, drop_path_rate=0.2)
    del encoder.classifier
    mean = encoder.default_cfg["mean"]
    std = encoder.default_cfg["std"]

    if pretrained:
        from torch.utils import model_zoo

        src_state_dict = model_zoo.load_url(
            "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth"
        )
        dst_state_dict = encoder.state_dict()

        for key, dst_tensor in dst_state_dict.items():
            dst_tensor_size = dst_tensor.size()
            if key in src_state_dict:
                src_tensor = src_state_dict[key]
                src_tensor_size = src_tensor.size()
                # If shape of tensors does not match, we pad necessary with random weights
                if src_tensor_size != dst_tensor_size:
                    assert len(src_tensor_size) == len(dst_tensor_size)
                    old = src_state_dict[key]
                    src_state_dict[key] = dst_tensor.clone()
                    slice_size = [slice(0, src_size) for src_size, dst_size in zip(src_tensor_size, dst_tensor_size)]
                    src_state_dict[key][slice_size] = old

        encoder.load_state_dict(src_state_dict, strict=False)

    return TimmRgbModel(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        input_key=INPUT_FEATURES_JPEG_FLOAT,
        need_embedding=need_embedding,
    )
