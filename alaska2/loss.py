import math

import torch
import torch.nn.functional as F
from catalyst.dl import AccuracyCallback
from catalyst.dl.callbacks import CriterionAggregatorCallback
from pytorch_toolbelt.losses import FocalLoss
from torch import nn

from .metric import CompetitionMetricCallback
from .dataset import *
from .cutmix import CutmixCallback
from .mixup import MixupCriterionCallback, MixupInputCallback
from .tsa import TSACriterionCallback

__all__ = ["OHEMCrossEntropyLoss", "ArcFaceLoss", "get_loss", "get_criterions", "get_criterion_callback"]


class OHEMCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Online hard example mining CE loss

    https://arxiv.org/pdf/1812.05802.pdf
    """

    def __init__(self, weight=None, fraction=0.3, ignore_index=-100, reduction="mean"):
        super().__init__(weight, ignore_index=ignore_index, reduction=reduction)
        self.fraction = fraction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(0)

        with torch.no_grad():
            positive_mask = (target > 0).view(batch_size, -1)
            Cp = torch.sum(positive_mask, dim=1)  # Number of positive pixels
            Cn = torch.sum(~positive_mask, dim=1)  # Number of negative pixels
            Chn = torch.max((Cn / 4).clamp_min(5), 2 * Cp)

        losses = F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction="none"
        ).view(target.size(0), -1)

        loss = 0
        num_samples = 0

        for i in range(batch_size):
            positive_losses = losses[i, positive_mask[i]]
            negative_losses = losses[i, ~positive_mask[i]]

            num_negatives = Chn[i]
            hard_negative_losses, _ = negative_losses.sort(descending=True)[:num_negatives]

            loss = positive_losses.sum() + hard_negative_losses.sum() + loss

            num_samples += positive_losses.size(0)
            num_samples += hard_negative_losses.size(0)

        loss /= float(num_samples)
        return loss


class ArcFaceLoss(nn.modules.Module):
    """
    https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109#latest-560973
    """

    def __init__(self, s=30.0, m=0.35, gamma=1, ignore_index=-100):
        super(ArcFaceLoss, self).__init__()
        self.gamma = gamma
        self.classify_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.s = s
        self.easy_margin = False
        self.cos_m = float(math.cos(m))
        self.sin_m = float(math.sin(m))
        self.th = float(math.cos(math.pi - m))
        self.mm = float(math.sin(math.pi - m) * m)

    def forward(self, cos_theta: torch.Tensor, labels):
        num_classes = cos_theta.size(1)
        sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        phi = (cos_theta * self.cos_m - sine * self.sin_m).type(cos_theta.dtype)
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = F.one_hot(labels, num_classes).type(cos_theta.dtype)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cos_theta, labels)

        loss = (loss1 + self.gamma * loss2) / (1 + self.gamma)
        return loss


def get_loss(loss_name: str, tsa=False):
    if loss_name.lower() == "bce":
        return nn.BCEWithLogitsLoss(reduction="none" if tsa else "mean")

    if loss_name.lower() == "ce":
        return nn.CrossEntropyLoss(reduction="none" if tsa else "mean")

    if loss_name.lower() == "focal":
        return FocalLoss(alpha=None, gamma=2, reduction="none" if tsa else "mean")

    if loss_name.lower() == "nfl":
        return FocalLoss(alpha=None, gamma=2, normalized=True, reduction="sum" if tsa else "mean")

    if loss_name.lower() == "ohem_ce":
        return OHEMCrossEntropyLoss()

    if loss_name.lower() == "arc_face":
        return ArcFaceLoss()

    raise KeyError(loss_name)


def get_criterion_callback(
    loss_name,
    input_key,
    output_key,
    num_epochs: int,
    prefix=None,
    loss_weight=1.0,
    mixup=False,
    cutmix=False,
    tsa=False,
):
    from catalyst.dl import CriterionCallback

    if prefix is None:
        prefix = f"{prefix}/{loss_name}"

    criterions_dict = {f"{prefix}/{loss_name}": get_loss(loss_name, tsa=tsa)}

    if mixup:
        criterion_callback = MixupCriterionCallback(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            criterion_key=f"{prefix}/{loss_name}",
            multiplier=float(loss_weight),
        )

    elif cutmix:
        criterion_callback = CutmixCallback(
            alpha=1.0,
            prefix=prefix,
            image_key=INPUT_IMAGE_KEY,
            input_key=input_key,
            output_key=output_key,
            criterion_key=f"{prefix}/{loss_name}",
            multiplier=float(loss_weight),
        )

    elif tsa:
        criterion_callback = TSACriterionCallback(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            criterion_key=f"{prefix}/{loss_name}",
            multiplier=float(loss_weight),
            num_classes=4,
            num_epochs=num_epochs,
        )

    else:
        criterion_callback = CriterionCallback(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            criterion_key=f"{prefix}/{loss_name}",
            multiplier=float(loss_weight),
        )

    return criterions_dict, criterion_callback, prefix


def get_criterions(
    modification_flag, modification_type, num_epochs: int, mixup=False, cutmix=False, tsa=False,
):
    criterions_dict = {}
    callbacks = []
    losses = []

    for criterion in modification_flag:
        if isinstance(criterion, (list, tuple)):
            loss_name, loss_weight = criterion
        else:
            loss_name, loss_weight = criterion, 1.0

        cd, criterion, criterion_name = get_criterion_callback(
            loss_name,
            num_epochs=num_epochs,
            input_key=INPUT_TRUE_MODIFICATION_FLAG,
            output_key=OUTPUT_PRED_MODIFICATION_FLAG,
            prefix=f"modification_flag/{loss_name}",
            loss_weight=float(loss_weight),
            mixup=mixup,
            cutmix=cutmix,
            tsa=tsa,
        )
        criterions_dict.update(cd)
        callbacks.append(criterion)
        losses.append(criterion_name)
        print("Using loss", loss_name, loss_weight)

    for criterion in modification_type:
        if isinstance(criterion, (list, tuple)):
            loss_name, loss_weight = criterion
        else:
            loss_name, loss_weight = criterion, 1.0

        cd, criterion, criterion_name = get_criterion_callback(
            loss_name,
            num_epochs=num_epochs,
            input_key=INPUT_TRUE_MODIFICATION_TYPE,
            output_key=OUTPUT_PRED_MODIFICATION_TYPE,
            prefix=f"modification_type/{loss_name}",
            loss_weight=float(loss_weight),
            mixup=mixup,
            cutmix=cutmix,
            tsa=tsa,
        )
        criterions_dict.update(cd)
        callbacks.append(criterion)
        losses.append(criterion_name)
        print("Using loss", loss_name, loss_weight)

    callbacks.append(CriterionAggregatorCallback(prefix="loss", loss_keys=losses))
    if mixup:
        callbacks.append(MixupInputCallback(fields=[INPUT_IMAGE_KEY], alpha=0.5, p=0.5),)

    if modification_flag is not None:
        callbacks.append(
            CompetitionMetricCallback(
                input_key=INPUT_TRUE_MODIFICATION_FLAG, output_key=OUTPUT_PRED_MODIFICATION_FLAG, prefix="auc"
            )
        )
    if modification_type is not None:
        callbacks.append(
            AccuracyCallback(
                input_key=INPUT_TRUE_MODIFICATION_TYPE, output_key=OUTPUT_PRED_MODIFICATION_TYPE, prefix="accuracy"
            )
        )
    return criterions_dict, callbacks