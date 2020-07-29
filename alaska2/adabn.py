import torch
from catalyst.utils import any2device
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = False


def fix_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = True


def bn_update(loader: DataLoader, model: nn.Module):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return

    assert loader.drop_last

    model.train()
    model.apply(reset_bn)

    for batch in tqdm(loader, desc="AdaBN"):
        batch = any2device(batch, device="cuda")
        model(**batch)

    model.apply(fix_bn)
