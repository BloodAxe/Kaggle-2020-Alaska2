import math

import torch
from catalyst.contrib.optimizers import RAdam, Lamb
from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.optimizer import Optimizer


# import torch.optim as Optimizer

# Original source:  https://github.com/shivram1987/diffGrad/blob/master/diffGrad.py

# modifications: @lessw2020


class DiffGrad(Optimizer):
    r"""Implements diffGrad algorithm. It is modified from the pytorch implementation of Adam.
    It has been proposed in `diffGrad: An Optimization Method for Convolutional Neural Networks`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _diffGrad: An Optimization Method for Convolutional Neural Networks:
        https://arxiv.org/abs/1909.11015
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DiffGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DiffGrad, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "diffGrad does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Previous gradient
                    state["previous_grad"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, previous_grad = state["exp_avg"], state["exp_avg_sq"], state["previous_grad"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # compute diffgrad coefficient (dfc)
                diff = abs(previous_grad - grad)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                state["previous_grad"] = grad

                # update momentum with dfc
                exp_avg1 = exp_avg * dfc

                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg1, denom)

        return loss


def _check_valid_opt_params(lr, eps, betas):
    if lr < 0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError(f"Betas have to be between 0 and 1: {betas}")


class Novograd(Optimizer):
    """Implements Novograd algorithm.
    It has been proposed  in "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks"
    (https://arxiv.org/abs/1905.11286)
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.95, 0.98),
        eps=1e-8,
        weight_decay=0,
        grad_averaging=False,
        amsgrad=False,
        luc=False,
        luc_trust=1e-3,
        luc_eps=1e-8,
    ):
        _check_valid_opt_params(lr, eps, betas)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging, amsgrad=amsgrad
        )
        self.luc = luc
        self.luc_trust = luc_trust
        self.luc_eps = luc_eps
        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                amsgrad = group["amsgrad"]
                state = self.state[p]

                # State initialization
                if not state:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)
                    if amsgrad:
                        # Maintains max of all exp moving avg of squared grad
                        state["max_exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                norm = grad.norm().pow(2)

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains max of all 2nd moment running avg till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                grad.div_(denom)
                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)
                if group["grad_averaging"]:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                if self.luc:
                    # Clip update so that updates are less than eta*weights
                    data_norm = torch.norm(p.data)
                    grad_norm = torch.norm(exp_avg.data)
                    luc_factor = self.luc_trust * data_norm / (grad_norm + self.luc_eps)
                    luc_factor = min(luc_factor, group["lr"])
                    p.data.add_(-luc_factor, exp_avg)
                else:
                    p.data.add_(-group["lr"], exp_avg)

        return loss


def get_optimizer(optimizer_name: str, parameters, learning_rate: float, weight_decay=1e-5, **kwargs):
    if optimizer_name.lower() == "sgd":
        return SGD(parameters, learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "adam":
        return Adam(parameters, learning_rate, weight_decay=weight_decay, eps=1e-5, **kwargs)  # As Jeremy suggests

    if optimizer_name.lower() == "rms":
        return RMSprop(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "adamw":
        return AdamW(parameters, learning_rate, weight_decay=weight_decay, eps=1e-5, **kwargs)

    if optimizer_name.lower() == "radam":
        return RAdam(parameters, learning_rate, weight_decay=weight_decay, eps=1e-5, **kwargs)  # As Jeremy suggests

    # if optimizer_name.lower() == "ranger":
    #     return Ranger(parameters, learning_rate, weight_decay=weight_decay,
    #                   **kwargs)

    # if optimizer_name.lower() == "qhadamw":
    #     return QHAdamW(parameters, learning_rate, weight_decay=weight_decay,
    #                    **kwargs)
    #
    if optimizer_name.lower() == "lamb":
        return Lamb(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "fused_lamb":
        from apex.optimizers import FusedLAMB

        return FusedLAMB(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "fused_adam":
        from apex.optimizers import FusedAdam

        return FusedAdam(parameters, learning_rate, eps=1e-5, weight_decay=weight_decay, adam_w_mode=True, **kwargs)

    if optimizer_name.lower() == "diffgrad":
        return DiffGrad(parameters, learning_rate, eps=1e-5, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "novograd":
        return Novograd(parameters, learning_rate, eps=1e-5, weight_decay=weight_decay, **kwargs)

    raise ValueError("Unsupported optimizer name " + optimizer_name)
