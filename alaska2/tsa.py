import math

import torch
from catalyst.dl import CriterionCallback, RunnerState
import torch.nn.functional as F

# from pytorch_toolbelt.utils.catalyst import TSACriterionCallback


class TSACriterionCallback(CriterionCallback):
    """
    Criterion callback with training signal annealing support.

    This callback requires that criterion key returns loss per each element in batch

    Reference:
        Unsupervised Data Augmentation for Consistency Training
        https://arxiv.org/abs/1904.12848
    """

    def __init__(
        self,
        num_classes,
        num_epochs,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        schedule="linear_schedule",
        multiplier: float = 1.0,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            multiplier=multiplier,
        )
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.tsa_threshold = None
        self.schedule = schedule

    def get_tsa_threshold(self, current_epoch, schedule, start, end):
        training_progress = float(current_epoch) / float(self.num_epochs)

        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
            # [exp(-5), exp(0)] = [1e-2, 1]
        elif schedule == "log_schedule":
            scale = 5
            # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
            threshold = 1 - math.exp((-training_progress) * scale)
        else:
            raise KeyError(schedule)

        return threshold * (end - start) + start

    def on_loader_start(self, state: RunnerState):
        if state.loader_name == "train":
            # self.tsa_threshold = self.get_tsa_threshold(
            #     state.epoch, self.schedule, start=1.0 / self.num_classes, end=1.0
            # )
            self.tsa_threshold = self.get_tsa_threshold(state.epoch, self.schedule, start=0.5, end=1.0)
            state.metrics.epoch_values[state.loader_name][f"{self.prefix}/tsa_threshold"] = self.tsa_threshold

    def _compute_loss_value(self, state: RunnerState, criterion):
        logits = self._get_output(state.output, self.output_key)
        targets = self._get_input(state.input, self.input_key)

        loss = criterion(logits, targets)

        if state.loader_name != "train":
            return loss.mean()

        with torch.no_grad():
            one_hot_targets = F.one_hot(targets.detach(), num_classes=self.num_classes).float()
            sup_probs = logits.detach().softmax(dim=1)
            correct_label_probs = torch.sum(one_hot_targets * sup_probs, dim=1)
            larger_than_threshold: torch.Tensor = correct_label_probs > self.tsa_threshold
            loss_mask: torch.Tensor = 1.0 - larger_than_threshold.float()

        loss = loss * loss_mask
        num_samples = loss_mask.sum()
        loss = loss.sum() / num_samples.clamp_min(1)

        state.metrics.add_batch_value(f"{self.prefix}/tsa_samples", int(num_samples))
        return loss
