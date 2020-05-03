from typing import List

import torch
from catalyst.dl import CriterionCallback, RunnerState
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    ar = W / float(H)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat * ar)
    cut_h = np.int(H * cut_rat / ar)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutmixCallback(CriterionCallback):
    """
    Callback to do cutmix augmentation.

    Paper: https://arxiv.org/abs/1710.09412

    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.

        You may not use them together.
    """

    def __init__(self, image_key: str = "features", alpha=1.0, p=0.5, on_train_only=True, **kwargs):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.image_key = image_key
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True
        self.is_batch_needed = True
        self.p = p

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        self.is_batch_needed = np.random.random() < self.p

        if not self.is_batch_needed:
            return

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        self.index = torch.randperm(state.input[self.image_key].shape[0])
        self.index.to(state.device)

        input = state.input[self.image_key]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        self.lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        input[:, :, bbx1:bbx2, bby1:bby2] = input[self.index, :, bbx1:bbx2, bby1:bby2].clone()

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed or not self.is_batch_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        loss = self.lam * criterion(pred, y_a) + (1 - self.lam) * criterion(pred, y_b)
        return loss
