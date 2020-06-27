import math

import torch
import torch.nn.functional as F
from catalyst.dl import AccuracyCallback
from catalyst.dl.callbacks import CriterionAggregatorCallback
from pytorch_toolbelt.losses import FocalLoss, BinaryFocalLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss
from pytorch_toolbelt.utils.catalyst import (
    BestMetricCheckpointCallback,
    ConfusionMatrixCallback,
    TrainOnlyCriterionCallback,
)
from torch import nn

from .cutmix import CutmixCallback
from .dataset import *
from .metric import *
from .mixup import MixupCriterionCallback, MixupInputCallback
from .tsa import TSACriterionCallback


__all__ = [
    "OHEMCrossEntropyLoss",
    "ArcFaceLoss",
    "PairwiseRankingLoss",
    "PairwiseRankingLossV2",
    "get_loss",
    "get_criterions",
    "get_criterion_callback",
]


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


class ContrastiveCosineEmbeddingLoss(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input: [N,E]
        """
        batch_size, embedding_size = input.size()

        binary_target = target > 0  # So far ignore particular type

        target_a = binary_target.view(batch_size, 1)
        target_b = binary_target.view(1, batch_size)

        same_class_mask = target_a == target_b

        input_a = input.view(batch_size, 1, embedding_size)
        input_b = input.view(1, batch_size, embedding_size)

        # Compute pairwise cosine similarity
        cossim = F.cosine_similarity(input_a, input_b, dim=2)

        # Two terms:
        # Inter-class variance should be minimized
        class_counts = torch.histc(target, 4)
        same_class_loss = 0

        for i, counts in enumerate(class_counts):
            if counts == 0:
                continue
            target_a = (target == i).view(batch_size, 1)
            target_b = (target == i).view(1, batch_size)
            same_class_mask = target_a == target_b

            single_class_loss: torch.Tensor = (1 - cossim) * same_class_mask.to(input.dtype)
            single_class_loss = single_class_loss.triu_(diagonal=1).sum() / counts

            same_class_loss = same_class_loss + single_class_loss

        # Distance between different classes should be maximized
        margin = 0.5
        diff_class_loss = F.relu(cossim * (~same_class_mask).to(input.dtype) - margin)
        diff_class_loss = diff_class_loss.triu_(diagonal=1).sum()

        return (same_class_loss + diff_class_loss) / batch_size


class EmbeddingLoss(nn.Module):
    """
    This loss assumes embedding vectors has length [N*4] and batch has elem1, elem1, elem2, elem3
    """

    def forward(self, input: torch.Tensor, _):
        """
        input: [N,E]
        """
        batch_size, embedding_size = input.size(0), input.size(1)
        num_unique_images = batch_size // 4
        is_image = len(input.size()) == 4

        loss = 0
        for i in range(num_unique_images):
            cover = input[i * 4 + 0]
            jmipod = input[i * 4 + 1]
            juniward = input[i * 4 + 2]
            uerd = input[i * 4 + 3]

            jmipod_loss = F.cosine_similarity(cover, jmipod, dim=0).pow_(2)
            juniward_loss = F.cosine_similarity(cover, juniward, dim=0).pow_(2)
            uerd_loss = F.cosine_similarity(cover, uerd, dim=0).pow_(2)

            if is_image:
                jmipod_loss = jmipod_loss.mean()
                juniward_loss = juniward_loss.mean()
                uerd_loss = uerd_loss.mean()

            sample_loss = jmipod_loss + juniward_loss + uerd_loss
            loss += sample_loss

        return loss / num_unique_images


class EmbeddingLossV2(nn.Module):
    """
    This loss assumes embedding vectors has length [N*4] and batch has elem1, elem1, elem2, elem3
    """

    def forward(self, input: torch.Tensor, t):
        """
        input: [N,E]
        """
        batch_size, embedding_size = input.size(0), input.size(1)
        num_unique_images = batch_size // 4
        is_image = len(input.size()) == 4

        background = torch.zeros(embedding_size, dtype=input.dtype, device=input.device)
        background[0] = 1
        margin = 0.1

        loss = 0
        for i in range(num_unique_images):
            cover = input[i * 4 + 0]
            jmipod = input[i * 4 + 1]
            juniward = input[i * 4 + 2]
            uerd = input[i * 4 + 3]

            # Attract unedited images to fixed embedding
            cover_loss = F.relu(1 - F.cosine_similarity(cover, background, dim=0).pow_(2) - margin)

            jmipod_loss = F.relu(F.cosine_similarity(cover, jmipod, dim=0).pow_(2) - margin)
            juniward_loss = F.relu(F.cosine_similarity(cover, juniward, dim=0).pow_(2) - margin)
            uerd_loss = F.relu(F.cosine_similarity(cover, uerd, dim=0).pow_(2) - margin)

            if is_image:
                cover_loss = cover_loss.mean()
                jmipod_loss = jmipod_loss.mean()
                juniward_loss = juniward_loss.mean()
                uerd_loss = uerd_loss.mean()

            sample_loss = cover_loss + (jmipod_loss + juniward_loss + uerd_loss) * 0.3333333
            loss += sample_loss

        return loss / num_unique_images


class PairwiseRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: [N,E]
        """
        batch_size = input.size(0)

        # View to [B,2], 0 are negatives, 1 are positives
        input = input.view(batch_size // 2, 2)
        target = target.view(batch_size)

        # cost = - torch.mean(
        #     torch.sigmoid(input @ torch.transpose(input)) * np.maximum(y @ np.ones(y.shape).T - np.ones(y.shape) @ y.T, 0)
        # )

        loss = -F.logsigmoid(input[:, 1] - input[:, 0]).mean()
        return loss


class PairwiseRankingLossV2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        input: [N,E]
        """

        y_pred = F.logsigmoid(y_pred)
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]

        loss = F.relu(0.2 + neg.mean() - pos.mean())
        return loss


class RocAucLoss(nn.Module):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """

    # https://github.com/tflearn/tflearn/blob/5a674b7f7d70064c811cbd98c4a41a17893d44ee/tflearn/objectives.py
    def forward(self, y_pred, y_true):
        eps = 1e-4
        y_pred = torch.sigmoid(y_pred).clamp(eps, 1 - eps)
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]

        pos = torch.unsqueeze(pos, 0)
        neg = torch.unsqueeze(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.7
        p = 2

        difference = torch.zeros_like(pos * neg) + pos - neg - gamma
        mask = difference > 0
        masked = difference.masked_fill(mask, 0)
        return torch.mean(torch.pow(-masked, p))


class RocAucLossCE(nn.Module):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """

    # https://github.com/tflearn/tflearn/blob/5a674b7f7d70064c811cbd98c4a41a17893d44ee/tflearn/objectives.py
    def forward(self, y_pred, y_true):
        eps = 1e-4
        y_pred = torch.softmax(y_pred, 1)[:, 1:].sum(dim=1)
        pos = y_pred[y_true > 0]
        neg = y_pred[y_true == 0]

        pos = torch.unsqueeze(pos, 0)
        neg = torch.unsqueeze(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3

        difference = torch.zeros_like(pos * neg) + pos - neg - gamma
        masked = difference[difference < 0.0]
        return torch.sum(torch.pow(-masked, p))


def get_loss(loss_name: str, tsa=False):
    if loss_name.lower() == "rank":
        return PairwiseRankingLoss()

    if loss_name.lower() == "rank2":
        return PairwiseRankingLossV2()

    if loss_name.lower() == "ccos":
        return ContrastiveCosineEmbeddingLoss()

    if loss_name.lower() == "cntr":
        return EmbeddingLoss()

    if loss_name.lower() == "roc_auc":
        return RocAucLoss()

    if loss_name.lower() == "roc_auc_ce":
        return RocAucLossCE()

    if loss_name.lower() == "cntrv2":
        return EmbeddingLossV2()

    if loss_name.lower() == "bce":
        return nn.BCEWithLogitsLoss(reduction="none" if tsa else "mean")

    if loss_name.lower() == "wbce":
        return nn.BCEWithLogitsLoss(reduction="none" if tsa else "mean", pos_weight=torch.tensor(0.33).float()).cuda()

    if loss_name.lower() == "wbce2":
        return nn.BCEWithLogitsLoss(reduction="none" if tsa else "mean", pos_weight=torch.tensor(0.66).float()).cuda()

    if loss_name.lower() == "ce":
        return nn.CrossEntropyLoss(reduction="none" if tsa else "mean")

    if loss_name.lower() == "soft_ce":
        return SoftCrossEntropyLoss(reduction="none" if tsa else "mean", smooth_factor=0.1)

    if loss_name.lower() == "soft_bce":
        return SoftBCEWithLogitsLoss(reduction="none" if tsa else "mean", smooth_factor=0.1, ignore_index=None)

    if loss_name.lower() == "wce":
        return nn.CrossEntropyLoss(
            reduction="none" if tsa else "mean", weight=torch.tensor([2, 1, 2, 1]).float()
        ).cuda()

    if loss_name.lower() == "focal":
        return FocalLoss(alpha=None, gamma=2, reduction="none" if tsa else "mean")

    if loss_name.lower() == "binary_focal":
        return BinaryFocalLoss(alpha=None, gamma=2, reduction="none" if tsa else "mean")

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
        if loss_name in {"rank", "rank2"}:
            criterion_cls = TrainOnlyCriterionCallback
        else:
            criterion_cls = CriterionCallback

        criterion_callback = criterion_cls(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            criterion_key=f"{prefix}/{loss_name}",
            multiplier=float(loss_weight),
        )

    return criterions_dict, criterion_callback, prefix


def get_criterions(
    modification_flag,
    modification_type,
    embedding_loss,
    num_epochs: int,
    feature_maps_loss=None,
    mixup=False,
    cutmix=False,
    tsa=False,
    class_names=["Cover", "JMiPOD", "JUNIWARD", "UERD"],
):
    criterions_dict = {}
    callbacks = []
    losses = []
    need_embedding_auc_score = False

    if modification_flag is not None:

        # Metrics
        callbacks += [
            ConfusionMatrixCallback(
                input_key=INPUT_TRUE_MODIFICATION_FLAG,
                output_key=OUTPUT_PRED_MODIFICATION_FLAG,
                prefix="flag",
                activation_fn=lambda x: x.sigmoid() > 0.5,
                class_names=["Original", "Modified"],
            ),
            CompetitionMetricCallback(
                input_key=INPUT_TRUE_MODIFICATION_TYPE,
                output_key=OUTPUT_PRED_MODIFICATION_FLAG,
                prefix="auc",
                output_activation=binary_logits_to_probas,
                class_names=class_names,
            ),
            OutputDistributionCallback(
                input_key=INPUT_TRUE_MODIFICATION_FLAG,
                output_key=OUTPUT_PRED_MODIFICATION_FLAG,
                output_activation=binary_logits_to_probas,
                prefix="distribution/binary",
            ),
            AccuracyCallback(
                prefix="accuracy_b",
                activation="Sigmoid",
                input_key=INPUT_TRUE_MODIFICATION_FLAG,
                output_key=OUTPUT_PRED_MODIFICATION_FLAG,
                accuracy_args=[1],
                num_classes=1,
            ),
            BestMetricCheckpointCallback(target_metric="auc", target_metric_minimize=False, save_n_best=3),
        ]

        # Losses
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

    if modification_type is not None:
        # Metrics
        callbacks += [
            ConfusionMatrixCallback(
                input_key=INPUT_TRUE_MODIFICATION_TYPE,
                output_key=OUTPUT_PRED_MODIFICATION_TYPE,
                prefix="type",
                class_names=class_names,
            ),
            AccuracyCallback(
                input_key=INPUT_TRUE_MODIFICATION_TYPE, output_key=OUTPUT_PRED_MODIFICATION_TYPE, prefix="accuracy"
            ),
            CompetitionMetricCallback(
                input_key=INPUT_TRUE_MODIFICATION_TYPE,
                output_key=OUTPUT_PRED_MODIFICATION_TYPE,
                output_activation=classifier_logits_to_probas,
                prefix="auc_classifier",
                class_names=class_names,
            ),
            OutputDistributionCallback(
                input_key=INPUT_TRUE_MODIFICATION_FLAG,
                output_key=OUTPUT_PRED_MODIFICATION_TYPE,
                output_activation=classifier_logits_to_probas,
                prefix="distribution/classifier",
            ),
            BestMetricCheckpointCallback(target_metric="auc_classifier", target_metric_minimize=False, save_n_best=3),
        ]

        # Losses
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

    if embedding_loss is not None:
        for criterion in embedding_loss:
            if isinstance(criterion, (list, tuple)):
                loss_name, loss_weight = criterion
            else:
                loss_name, loss_weight = criterion, 1.0

            cd, criterion, criterion_name = get_criterion_callback(
                loss_name,
                num_epochs=num_epochs,
                input_key=INPUT_TRUE_MODIFICATION_TYPE,
                output_key=OUTPUT_PRED_EMBEDDING,
                prefix=f"embedding/{loss_name}",
                loss_weight=float(loss_weight),
                mixup=mixup,
                cutmix=cutmix,
                tsa=tsa,
            )
            criterions_dict.update(cd)
            callbacks.append(criterion)
            losses.append(criterion_name)
            print("Using loss", loss_name, loss_weight)

            if loss_name == "cntrv2":
                need_embedding_auc_score = True

        if need_embedding_auc_score:
            callbacks += [
                OutputDistributionCallback(
                    input_key=INPUT_TRUE_MODIFICATION_FLAG,
                    output_key=OUTPUT_PRED_EMBEDDING,
                    output_activation=embedding_to_probas,
                    prefix="distribution/embedding",
                ),
                CompetitionMetricCallback(
                    input_key=INPUT_TRUE_MODIFICATION_TYPE,
                    output_key=OUTPUT_PRED_EMBEDDING,
                    prefix="auc_embedding",
                    output_activation=embedding_to_probas,
                    class_names=class_names,
                ),
            ]

    if feature_maps_loss is not None:
        for criterion in feature_maps_loss:
            if isinstance(criterion, (list, tuple)):
                loss_name, loss_weight = criterion
            else:
                loss_name, loss_weight = criterion, 1.0

            for fm in {OUTPUT_FEATURE_MAP_4, OUTPUT_FEATURE_MAP_8, OUTPUT_FEATURE_MAP_16, OUTPUT_FEATURE_MAP_32}:
                cd, criterion, criterion_name = get_criterion_callback(
                    loss_name,
                    num_epochs=num_epochs,
                    input_key=INPUT_TRUE_MODIFICATION_TYPE,
                    output_key=fm,
                    prefix=f"{fm}/{loss_name}",
                    loss_weight=float(loss_weight),
                    mixup=mixup,
                    cutmix=cutmix,
                    tsa=tsa,
                )

                criterions_dict.update(cd)
                callbacks.append(criterion)
                losses.append(criterion_name)
                print("Using loss", fm, loss_name, loss_weight)

    callbacks.append(CriterionAggregatorCallback(prefix="loss", loss_keys=losses))
    if mixup:
        callbacks.append(MixupInputCallback(fields=[INPUT_IMAGE_KEY], alpha=0.5, p=0.5))

    return criterions_dict, callbacks
