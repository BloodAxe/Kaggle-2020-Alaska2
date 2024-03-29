import numpy as np
import torch
from pytorch_toolbelt.modules import Normalize, GlobalAvgPool2d
from torch import nn

from alaska2.dataset import INPUT_IMAGE_KEY, OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_MODIFICATION_TYPE
from alaska2.models.modules import TLU, SqrtmLayer, CovpoolLayer, TriuvecLayer
from alaska2.models.srm_filter_kernel import all_normalized_hpf_list

__all__ = ["HPFNet", "hpf_net_v2", "hpf_net", "hpf_b3_fixed_covpool", "hpf_b3_fixed_gap", "hpf_b3_covpool"]


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode="constant")

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)

        return output


class HPF3(nn.Module):
    def __init__(self, trainable_hpf=False, stride=1):
        super(HPF3, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode="constant")

            all_hpf_list_5x5.append(hpf_item)

        features = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5)
        features = torch.cat([features, features, features], dim=1)

        hpf_weight = nn.Parameter(features, requires_grad=trainable_hpf)

        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, stride=stride, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)

        return output


class HPFNet(nn.Module):
    def __init__(self, num_classes, dropout=0, pretrained=False, trainable_hpf=False):
        super(HPFNet, self).__init__()

        max_pixel_value = 255
        self.rgb_bn = Normalize(
            [0.3914976 * max_pixel_value, 0.44266784 * max_pixel_value, 0.46043398 * max_pixel_value],
            [0.17819773 * max_pixel_value, 0.17319807 * max_pixel_value, 0.18128773 * max_pixel_value],
        )

        self.group1 = HPF3(trainable_hpf=trainable_hpf)

        self.group2 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)
        self.drop = nn.Dropout(dropout)
        features = int(256 * (256 + 1) / 2)
        self.type_classifier = nn.Linear(features, num_classes)
        self.flag_classifier = nn.Linear(features, 1)

    def forward(self, **kwargs):
        rgb = self.rgb_bn(kwargs[INPUT_IMAGE_KEY].float())

        output = self.group1(rgb)
        output = self.group2(output)
        output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)

        # Global covariance pooling
        output = CovpoolLayer(output)
        output = SqrtmLayer(output, 5)
        output = TriuvecLayer(output)

        x = output.view(output.size(0), -1)

        return {
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY]


class HPFNetCovPool(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)

        self.encoder = encoder
        self.drop = nn.Dropout(dropout)

        features = int(256 * (256 + 1) / 2)
        self.bottleneck = nn.Conv2d(encoder.num_features, 256, kernel_size=1)

        self.type_classifier = nn.Linear(features, num_classes)
        self.flag_classifier = nn.Linear(features, 1)

    def forward(self, **kwargs):
        x = self.rgb_bn(kwargs[INPUT_IMAGE_KEY])
        x = self.encoder.forward_features(x)
        x = self.bottleneck(x)

        # Global covariance pooling
        output = CovpoolLayer(x)
        output = SqrtmLayer(output, 5)
        output = TriuvecLayer(output)

        x = output.view(output.size(0), -1)

        return {
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY]


class HPFNetGAP(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        max_pixel_value = 255
        self.rgb_bn = Normalize(np.array(mean) * max_pixel_value, np.array(std) * max_pixel_value)

        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.pool = GlobalAvgPool2d(flatten=True)

        self.type_classifier = nn.Linear(encoder.num_features, num_classes)
        self.flag_classifier = nn.Linear(encoder.num_features, 1)

    def forward(self, **kwargs):
        x = self.rgb_bn(kwargs[INPUT_IMAGE_KEY])
        x = self.encoder.forward_features(x)
        x = self.pool(x)

        return {
            OUTPUT_PRED_MODIFICATION_TYPE: self.type_classifier(self.drop(x)),
            OUTPUT_PRED_MODIFICATION_FLAG: self.flag_classifier(self.drop(x)),
        }

    @property
    def required_features(self):
        return [INPUT_IMAGE_KEY]


def hpf_net(num_classes, dropout=0, pretrained=False):
    return HPFNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def hpf_b3_fixed_covpool(num_classes, dropout=0, pretrained=False):
    from timm.models import efficientnet

    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=True, drop_path_rate=0.1)
    encoder.conv_stem = nn.Sequential(HPF3(trainable_hpf=False, stride=2), nn.Conv2d(30, 40, kernel_size=1))
    del encoder.classifier

    return HPFNetCovPool(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def hpf_b3_covpool(num_classes, dropout=0, pretrained=False):
    from timm.models import efficientnet

    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=True, drop_path_rate=0.1)
    encoder.conv_stem = nn.Sequential(HPF3(trainable_hpf=True, stride=2), nn.Conv2d(30, 40, kernel_size=1))
    del encoder.classifier

    return HPFNetCovPool(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def hpf_b3_fixed_gap(num_classes, dropout=0, pretrained=False):
    from timm.models import efficientnet

    encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=True, drop_path_rate=0.1)
    encoder.conv_stem = nn.Sequential(HPF3(trainable_hpf=False, stride=2), nn.Conv2d(30, 40, kernel_size=1))
    del encoder.classifier

    return HPFNetGAP(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
        mean=encoder.default_cfg["mean"],
        std=encoder.default_cfg["std"],
    )


def hpf_net_v2(num_classes, dropout=0, pretrained=False):
    return HPFNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained, trainable_hpf=True)
