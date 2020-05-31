from typing import Callable

import torch
from catalyst.dl import Callback, RunnerState, CallbackOrder
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from sklearn import metrics
import numpy as np
import torch.nn.functional as F

__all__ = [
    "CompetitionMetricCallback",
    "alaska_weighted_auc",
    "OutputDistributionCallback",
    "binary_logits_to_probas",
    "classifier_logits_to_probas",
    "embedding_to_probas",
]

from alaska2 import du


def alaska_weighted_auc(y_true, y_pred):
    try:
        tpr_thresholds = [0.0, 0.4, 1.0]
        weights = [2, 1]

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)

        # size of subsets
        areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

        # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
        normalization = np.dot(areas, weights)

        competition_metric = 0
        for idx, weight in enumerate(weights):
            y_min = tpr_thresholds[idx]
            y_max = tpr_thresholds[idx + 1]
            mask = (y_min < tpr) & (tpr < y_max)

            if mask.sum() == 0:
                continue

            x_padding = np.linspace(fpr[mask][-1], 1, 100)
            x = np.concatenate([fpr[mask], x_padding])
            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

            y = y - y_min  # normalize such that curve starts at y=0
            score = metrics.auc(x, y)
            submetric = score * weight
            best_subscore = (y_max - y_min) * weight
            competition_metric += submetric

        return competition_metric / normalization
    except Exception as e:
        print(e)
        return 0


def binary_logits_to_probas(x):
    return x.sigmoid().squeeze(1)


def classifier_logits_to_probas(x):
    return x.softmax(dim=1)[:, 1:].sum(dim=1)


def embedding_to_probas(x: torch.Tensor):
    background = torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
    background[0] = 1

    predicted = 1 - F.cosine_similarity(x, background.unsqueeze(0), dim=1).pow_(2)
    return predicted


class CompetitionMetricCallback(Callback):
    def __init__(self, input_key: str, output_key: str, output_activation: Callable, prefix="auc", distributed=False):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.true_labels = []
        self.pred_labels = []
        self.output_activation = output_activation
        self.distributed = distributed

    def on_loader_start(self, state: RunnerState):
        self.true_labels = []
        self.pred_labels = []

    @torch.no_grad()
    def on_batch_end(self, state: RunnerState):
        output = self.output_activation(state.output[self.output_key].detach().cpu())
        self.true_labels.extend(to_numpy(state.input[self.input_key]).flatten())
        self.pred_labels.extend(to_numpy(output).flatten())

    def on_loader_end(self, state: RunnerState):
        true_labels = np.array(self.true_labels)
        pred_labels = np.array(self.pred_labels)

        if self.distributed:
            true_labels = du.all_gather(true_labels)
            true_labels = np.concatenate(true_labels)

            pred_labels = du.all_gather(pred_labels)
            pred_labels = np.concatenate(pred_labels)

        score = alaska_weighted_auc(true_labels, pred_labels)
        state.metrics.epoch_values[state.loader_name][self.prefix] = float(score)

        logger = get_tensorboard_logger(state)
        logger.add_pr_curve(self.prefix, true_labels, pred_labels)


class OutputDistributionCallback(Callback):
    def __init__(self, input_key: str, output_key: str, output_activation: Callable, prefix="distribution"):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.true_labels = []
        self.pred_labels = []
        self.output_activation = output_activation

    def on_loader_start(self, state: RunnerState):
        self.true_labels = []
        self.pred_labels = []

    @torch.no_grad()
    def on_batch_end(self, state: RunnerState):
        output = state.output[self.output_key].detach()

        self.true_labels.extend(to_numpy(state.input[self.input_key]).flatten())
        self.pred_labels.extend(to_numpy(self.output_activation(output)).flatten())

    def on_loader_end(self, state: RunnerState):
        true_labels = np.array(self.true_labels)
        pred_probas = np.array(self.pred_labels)

        logger = get_tensorboard_logger(state)
        logger.add_histogram(self.prefix + "/neg", pred_probas[true_labels == 0], state.epoch)
        logger.add_histogram(self.prefix + "/pos", pred_probas[true_labels == 1], state.epoch)
