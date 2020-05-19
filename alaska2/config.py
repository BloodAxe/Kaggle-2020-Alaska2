import collections
import gc
import json
import os
from datetime import datetime
from typing import Union, List
import jsonpickle

import torch
from pytorch_toolbelt.optimization.functional import freeze_model

from catalyst.dl import OptimizerCallback, SchedulerCallback, SupervisedRunner
from catalyst.utils import unpack_checkpoint, load_checkpoint
from pytorch_toolbelt.optimization.functional import get_optimizable_parameters
from pytorch_toolbelt.utils import count_parameters, fs, transfer_weights
from pytorch_toolbelt.utils.catalyst import (
    HyperParametersCallback,
    ShowPolarBatchesCallback,
    clean_checkpoint,
    report_checkpoint,
)
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from alaska2 import get_criterions, get_optimizer, get_scheduler, draw_predictions, get_model
from alaska2.dataset import get_datasets
from alaska2.models.timm import TimmRgbModel
from alaska2.models.ycrcb import YCrCbModel


class StageConfig:
    def __init__(self):
        self.stage_name = "main"

        # Optimizer settings
        self.optimizer = "RAdam"
        self.weight_decay = 0
        self.learning_rate = 1e-3
        self.schedule = None
        self.fp16 = False
        self.accumulation_steps = 1

        # Stage settings
        self.epochs = 1
        self.verbose = True
        self.experiment = None
        self.transfer_weights = None
        self.checkpoint_weights = None
        self.show = False

        # Model settings
        self.freeze_bn = False

        # Loss function settings
        self.mixup = False
        self.cutmix = False
        self.tsa = False
        self.modification_flag_loss = None
        self.modification_type_loss = None
        self.embedding_loss = None
        self.feature_maps_loss = None
        self.main_metric = "loss"
        self.main_metric_minimize = True

        # Data settings
        self.train_batch_size = 1
        self.valid_batch_size = 1
        self.negative_image_dir = None
        self.image_size = 512, 512
        self.augmentations = "light"
        self.obliterate_p = 0
        self.fast = False
        self.balance = False

        # This parameter controls whether to restore model state for best checkpoint from this stage
        self.restore_best = False


class ExperimenetConfig:
    def __init__(self):
        self.fold = None
        self.data_dir = os.environ.get("KAGGLE_2020_ALASKA2")
        self.num_workers = 0
        self.resume_from_checkpoint = None
        self.transfer_from_checkpoint = None

        self.stages: List[StageConfig] = []

        # Model
        self.model_name = None
        self.dropout = 0


def make_experiment_name(exp_config: ExperimenetConfig):
    current_time = datetime.now().strftime("%b%d_%H_%M")
    checkpoint_prefix = f"{current_time}_{exp_config.model_name}_fold{exp_config.fold}"
    return checkpoint_prefix


def run_whole_training(experiment_name: str, exp_config: ExperimenetConfig, runs_dir="runs"):
    model = get_model(exp_config.model_name, dropout=exp_config.dropout).cuda()

    if exp_config.transfer_from_checkpoint:
        transfer_checkpoint = fs.auto_file(exp_config.transfer_from_checkpoint)
        print("Transferring weights from model checkpoint", transfer_checkpoint)
        checkpoint = load_checkpoint(transfer_checkpoint)
        pretrained_dict = checkpoint["model_state_dict"]

        transfer_weights(model, pretrained_dict)

    if exp_config.resume_from_checkpoint:
        checkpoint = load_checkpoint(fs.auto_file(exp_config.resume_from_checkpoint))
        unpack_checkpoint(checkpoint, model=model)

        print("Loaded model weights from:", exp_config.resume_from_checkpoint)
        report_checkpoint(checkpoint)

    experiment_dir = os.path.join(runs_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=False)

    config_fname = os.path.join(experiment_dir, f"config.json")
    with open(config_fname, "w") as f:
        f.write(json.dumps(jsonpickle.encode(exp_config), indent=2))

    for stage in exp_config.stages:
        run_stage_training(model, stage, exp_config, experiment_dir=experiment_dir)


def run_stage_training(
    model: Union[TimmRgbModel, YCrCbModel], config: StageConfig, exp_config: ExperimenetConfig, experiment_dir: str
):
    # Preparing model
    freeze_model(model, freeze_bn=config.freeze_bn)

    train_ds, valid_ds, train_sampler = get_datasets(
        data_dir=exp_config.data_dir,
        image_size=config.image_size,
        augmentation=config.augmentations,
        balance=config.balance,
        fast=config.fast,
        fold=exp_config.fold,
        features=model.required_features,
        obliterate_p=config.obliterate_p,
    )

    criterions_dict, loss_callbacks = get_criterions(
        modification_flag=config.modification_flag_loss,
        modification_type=config.modification_type_loss,
        embedding_loss=config.embedding_loss,
        feature_maps_loss=config.feature_maps_loss,
        num_epochs=config.epochs,
        mixup=config.mixup,
        cutmix=config.cutmix,
        tsa=config.tsa,
    )

    callbacks = loss_callbacks + [
        OptimizerCallback(accumulation_steps=config.accumulation_steps, decouple_weight_decay=False),
        HyperParametersCallback(
            hparam_dict={
                "model": exp_config.model_name,
                "scheduler": config.schedule,
                "optimizer": config.optimizer,
                "augmentations": config.augmentations,
                "size": config.image_size[0],
                "weight_decay": config.weight_decay,
            }
        ),
    ]

    if config.show:
        callbacks += [ShowPolarBatchesCallback(draw_predictions, metric="loss", minimize=True)]

    loaders = collections.OrderedDict()
    loaders["train"] = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        num_workers=exp_config.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )

    loaders["valid"] = DataLoader(
        valid_ds, batch_size=config.valid_batch_size, num_workers=exp_config.num_workers, pin_memory=True
    )

    print("Stage            :", config.stage_name)
    print("  FP16 mode      :", config.fp16)
    print("  Fast mode      :", config.fast)
    print("  Epochs         :", config.epochs)
    print("  Workers        :", exp_config.num_workers)
    print("  Data dir       :", exp_config.data_dir)
    print("  Experiment dir :", experiment_dir)
    print("Data              ")
    print("  Augmentations  :", config.augmentations)
    print("  Obliterate (%) :", config.obliterate_p)
    print("  Negative images:", config.negative_image_dir)
    print("  Train size     :", len(loaders["train"]), "batches", len(train_ds), "samples")
    print("  Valid size     :", len(loaders["valid"]), "batches", len(valid_ds), "samples")
    print("  Image size     :", config.image_size)
    print("  Balance        :", config.balance)
    print("  Mixup          :", config.mixup)
    print("  CutMix         :", config.cutmix)
    print("  TSA            :", config.tsa)
    print("Model            :", exp_config.model_name)
    print("  Parameters     :", count_parameters(model))
    print("  Dropout        :", exp_config.dropout)
    print("Optimizer        :", config.optimizer)
    print("  Learning rate  :", config.learning_rate)
    print("  Weight decay   :", config.weight_decay)
    print("  Scheduler      :", config.schedule)
    print("  Batch sizes    :", config.train_batch_size, config.valid_batch_size)
    print("Losses            ")
    print("  Flag           :", config.modification_flag_loss)
    print("  Type           :", config.modification_type_loss)
    print("  Embedding      :", config.embedding_loss)
    print("  Feature maps   :", config.feature_maps_loss)

    optimizer = get_optimizer(
        config.optimizer,
        get_optimizable_parameters(model),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler(
        config.schedule,
        optimizer,
        lr=config.learning_rate,
        num_epochs=config.epochs,
        batches_in_epoch=len(loaders["train"]),
    )
    if isinstance(scheduler, CyclicLR):
        callbacks += [SchedulerCallback(mode="batch")]

    # model training
    runner = SupervisedRunner(input_key=model.required_features, output_key=None)
    runner.train(
        fp16=config.fp16,
        model=model,
        criterion=criterions_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        loaders=loaders,
        logdir=os.path.join(experiment_dir, config.stage_name),
        num_epochs=config.epochs,
        verbose=config.verbose,
        main_metric=config.main_metric,
        minimize_metric=config.main_metric_minimize,
        checkpoint_data={"config": config},
    )

    del optimizer, loaders, callbacks, runner

    best_checkpoint = os.path.join(experiment_dir, config.stage_name, "checkpoints", "best.pth")
    model_checkpoint = os.path.join(experiment_dir, f"{exp_config.checkpoint_prefix}.pth")
    clean_checkpoint(best_checkpoint, model_checkpoint)

    # Restore state of best model
    if config.restore_best:
        unpack_checkpoint(load_checkpoint(model_checkpoint), model=model)

    # Some memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
