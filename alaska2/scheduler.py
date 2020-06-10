import math
import warnings

from catalyst.contrib.schedulers import OneCycleLRWithWarmup
from pytorch_toolbelt.optimization.lr_schedules import PolyLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CyclicLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    _LRScheduler,
)


class PolyUpLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, max_epoch, gamma=0.9):
        def poly_lr(epoch):
            return (float(epoch + 1) / max_epoch) ** gamma

        super().__init__(optimizer, poly_lr)


class FlatCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        T_{cur} \neq (2k+1)T_{max};\\
        \eta_{t+1} = \eta_{t} + (\eta_{max} - \eta_{min})\frac{1 -
        \cos(\frac{1}{T_{max}}\pi)}{2},
        T_{cur} = (2k+1)T_{max}.\\

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_flat, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_flat = T_flat
        self.eta_min = eta_min
        super(FlatCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                DeprecationWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs
        elif (max(0, self.last_epoch - self.T_flat) - 1 - max(0, self.T_max - self.T_flat)) % (
            2 * max(0, self.T_max - self.T_flat)
        ) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / max(0, self.T_max - self.T_flat))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * max(0, self.last_epoch - self.T_flat) / max(0, self.T_max - self.T_flat)))
            / (1 + math.cos(math.pi * (max(0, self.last_epoch - self.T_flat) - 1) / max(0, self.T_max - self.T_flat)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * max(0, self.last_epoch - self.T_flat) / max(0, self.T_max - self.T_flat)))
            / 2
            for base_lr in self.base_lrs
        ]


def get_scheduler(scheduler_name: str, optimizer, lr, num_epochs, batches_in_epoch=None):
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    if scheduler_name.lower() == "poly_up":
        return PolyUpLR(optimizer, num_epochs, gamma=0.5)

    if scheduler_name.lower() == "cos":
        return CosineAnnealingLR(optimizer, num_epochs, eta_min=lr * 0.05)

    if scheduler_name.lower() == "flat_cos":
        return FlatCosineAnnealingLR(optimizer, num_epochs, int(num_epochs * 0.6), eta_min=5e-5)

    if scheduler_name.lower() == "flat_cos2":
        return FlatCosineAnnealingLR(optimizer, num_epochs, int(num_epochs * 0.5), eta_min=lr * 0.05)

    if scheduler_name.lower() == "cosr":
        return CosineAnnealingWarmRestarts(optimizer, T_0=max(2, num_epochs // 10), eta_min=5e-5)

    if scheduler_name.lower() in {"1cycle", "one_cycle"}:
        return OneCycleLRWithWarmup(
            optimizer, lr_range=(lr, 1e-6, 1e-5), num_steps=batches_in_epoch, warmup_fraction=0.05, decay_fraction=0.1
        )

    if scheduler_name.lower() == "exp":
        return ExponentialLR(optimizer, gamma=0.95)

    if scheduler_name.lower() == "clr":
        return CyclicLR(
            optimizer,
            base_lr=5e-5,
            max_lr=lr,
            step_size_up=batches_in_epoch // 4,
            # mode='exp_range',
            gamma=0.99,
        )

    if scheduler_name.lower() == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=[
                int(num_epochs * 0.1),
                int(num_epochs * 0.2),
                int(num_epochs * 0.3),
                int(num_epochs * 0.4),
                int(num_epochs * 0.5),
                int(num_epochs * 0.6),
                int(num_epochs * 0.7),
                int(num_epochs * 0.8),
                int(num_epochs * 0.9),
            ],
            gamma=0.9,
        )

    if scheduler_name.lower() == "simple":
        return MultiStepLR(optimizer, milestones=[int(num_epochs * 0.3), int(num_epochs * 0.6)], gamma=0.1)

    raise KeyError(scheduler_name)
