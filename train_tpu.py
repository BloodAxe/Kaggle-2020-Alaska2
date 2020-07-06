import pickle

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

# warnings.filterwarnings("ignore")

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from catalyst.utils import any2device
from pytorch_toolbelt.optimization.functional import get_optimizable_parameters
from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.distributed import all_gather
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import operator
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Resize
from efficientnet_pytorch import EfficientNet
from transformers import AdamW, get_cosine_schedule_with_warmup
from albumentations import *
from albumentations.pytorch import ToTensor
from tqdm import tqdm
import json
import time


from alaska2 import *
from alaska2.submissions import classifier_probas


def xla_all_gather(data, device):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    import torch_xla.core.xla_model

    world_size = xm.xrt_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    xla_model.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    xla_model.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def _run(
    model: nn.Module,
    prefix: str,
    data_dir: str,
    fold: int,
    epochs: int,
    batch_size: int,
    optimizer_name: str,
    augmentations="light",
    learning_rate=1e-4,
    weight_decay=0,
    fast=False,
):
    def train_fn(epoch, train_dataloader, optimizer, criterion, scheduler, device):
        model.train()

        for batch_idx, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch_data = any2device(batch_data, device)
            outputs = model(**batch_data)

            y_pred = outputs[OUTPUT_PRED_MODIFICATION_TYPE]
            y_true = batch_data[INPUT_TRUE_MODIFICATION_TYPE]

            loss = criterion(y_pred, y_true)

            if batch_idx % 100:
                xm.master_print(f"Batch: {batch_idx}, loss: {loss.item()}")

            loss.backward()
            xm.optimizer_step(optimizer)

            if scheduler is not None:
                scheduler.step()

    def valid_fn(epoch, valid_dataloader, criterion, device):
        model.eval()

        pred_scores = []
        true_scores = []

        for batch_idx, batch_data in enumerate(valid_dataloader):
            batch_data = any2device(batch_data, device)
            outputs = model(**batch_data)

            y_pred = outputs[OUTPUT_PRED_MODIFICATION_TYPE]
            y_true = batch_data[INPUT_TRUE_MODIFICATION_TYPE]

            loss = criterion(y_pred, y_true)

            pred_scores.extend(to_numpy(classifier_probas(y_pred)))
            true_scores.extend(to_numpy(y_true))

            xm.master_print(f"Batch: {batch_idx}, loss: {loss.item()}")

        val_wauc = alaska_weighted_auc(xla_all_gather(true_scores, device), xla_all_gather(pred_scores, device))
        xm.master_print(f"Valid epoch: {epoch}, wAUC: {val_wauc}")
        return val_wauc

    train_dataset, valid_dataset, _ = get_datasets(
        data_dir, fold=fold, fast=fast, augmentation=augmentations, features=model.required_features
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1, drop_last=False
    )

    device = xm.xla_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        optimizer_name, get_optimizable_parameters(model), learning_rate=learning_rate, weight_decay=weight_decay
    )
    num_train_steps = int(len(train_dataset) / batch_size / xm.xrt_world_size() * epochs)
    xm.master_print(f"num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}")

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True, min_lr=1e-6)

    best_wauc = 0

    train_begin = time.time()
    for epoch in range(epochs):
        para_loader = pl.ParallelLoader(train_dataloader, [device])

        start = time.time()
        print("*" * 15)
        print(f"EPOCH: {epoch + 1}")
        print("*" * 15)

        print("Training.....")
        train_fn(
            epoch=epoch + 1,
            train_dataloader=para_loader.per_device_loader(device),
            optimizer=optimizer,
            criterion=criterion,
            scheduler=None,
            device=device,
        )

        with torch.no_grad():
            para_loader = pl.ParallelLoader(valid_dataloader, [device])

            print("Validating....")
            val_wauc = valid_fn(
                epoch=epoch + 1,
                valid_dataloader=para_loader.per_device_loader(device),
                criterion=criterion,
                device=device,
            )

            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(val_wauc)

            xm.save(model.state_dict(), f"{prefix}_last.pth")
            if val_wauc > best_wauc:
                best_wauc = val_wauc
                xm.save(model.state_dict(), f"{prefix}_best.pth")
                xm.master_print(f"Saved best checkpoint with wAUC {best_wauc}")

        print(f"Epoch completed in {(time.time() - start) / 60} minutes")
    print(f"Training completed in {(time.time() - train_begin) / 60} minutes")


_run(
    model=get_model("rgb_tf_efficientnet_b6_ns", 4, dropout=0.1),
    prefix="",
    data_dir=DATA_DIR,
    fold=0,
    epochs=50,
    batch_size=16,
    optimizer_name="Ranger",
    fast=True,
)
