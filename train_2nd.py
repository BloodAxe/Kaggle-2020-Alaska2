from __future__ import absolute_import

import argparse
import collections
import gc
import json
import os
from datetime import datetime

import numpy as np
from catalyst.dl import SupervisedRunner, OptimizerCallback, SchedulerCallback
from pytorch_toolbelt.optimization.functional import get_optimizable_parameters
from pytorch_toolbelt.utils.catalyst import clean_checkpoint, HyperParametersCallback
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn
from torch.utils.data import DataLoader, Dataset

from alaska2 import *
from alaska2.models.stacker import StackingModel

INPUT_EMBEDDING_KEY = "input_embedding"


class StackerDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]

        return {
            INPUT_EMBEDDING_KEY: x,
            INPUT_TRUE_MODIFICATION_FLAG: torch.tensor([y]).float(),
            INPUT_TRUE_MODIFICATION_TYPE: int(y),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-acc", "--accumulation-steps", type=int, default=1, help="Number of batches to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--obliterate", type=float, default=0, help="Change of obliteration")
    parser.add_argument("-nid", "--negative-image-dir", type=str, default=None, help="Change of obliteration")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch Size during training, e.g. -b 64")
    parser.add_argument(
        "-wbs", "--warmup-batch-size", type=int, default=None, help="Batch Size during training, e.g. -b 64"
    )
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epoch to run")
    parser.add_argument(
        "-es", "--early-stopping", type=int, default=None, help="Maximum number of epochs without improvement"
    )
    parser.add_argument("-fe", "--freeze-encoder", action="store_true", help="Freeze encoder parameters for N epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Initial learning rate")

    parser.add_argument(
        "-l", "--modification-flag-loss", type=str, default=None, action="append", nargs="+"  # [["ce", 1.0]],
    )
    parser.add_argument(
        "--modification-type-loss", type=str, default=None, action="append", nargs="+"  # [["ce", 1.0]],
    )
    parser.add_argument("--embedding-loss", type=str, default=None, action="append", nargs="+")  # [["ce", 1.0]],
    parser.add_argument("--feature-maps-loss", type=str, default=None, action="append", nargs="+")  # [["ce", 1.0]],
    parser.add_argument("--mask-loss", type=str, default=None, action="append", nargs="+")  # [["ce", 1.0]],
    parser.add_argument("--bits-loss", type=str, default=None, action="append", nargs="+")  # [["ce", 1.0]],

    parser.add_argument("-o", "--optimizer", default="RAdam", help="Name of the optimizer")
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="Checkpoint filename to use as initial model weights"
    )
    parser.add_argument("-w", "--workers", default=8, type=int, help="Num workers")
    parser.add_argument("-a", "--augmentations", default="safe", type=str, help="Level of image augmentations")
    parser.add_argument("--transfer", default=None, type=str, help="")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--tsa", action="store_true")
    parser.add_argument("--fold", default=None, type=int)
    parser.add_argument("-s", "--scheduler", default=None, type=str, help="")
    parser.add_argument("-x", "--experiment", default=None, type=str, help="")
    parser.add_argument("-d", "--dropout", default=None, type=float, help="Dropout before head layer")
    parser.add_argument(
        "--warmup", default=0, type=int, help="Number of warmup epochs with reduced LR on encoder parameters"
    )
    parser.add_argument(
        "--fine-tune", default=0, type=int, help="Number of warmup epochs with reduced LR on encoder parameters"
    )
    parser.add_argument("-wd", "--weight-decay", default=0, type=float, help="L2 weight decay")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--freeze-bn", action="store_true")

    args = parser.parse_args()
    set_manual_seed(args.seed)

    assert (
        args.modification_flag_loss or args.modification_type_loss or args.embedding_loss
    ), "At least one of losses must be set"

    modification_flag_loss = args.modification_flag_loss
    modification_type_loss = args.modification_type_loss
    embedding_loss = args.embedding_loss
    feature_maps_loss = args.feature_maps_loss
    mask_loss = args.mask_loss
    bits_loss = args.bits_loss

    data_dir = args.data_dir
    cache = args.cache
    num_workers = args.workers
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    optimizer_name = args.optimizer
    fast = args.fast
    augmentations = args.augmentations
    fp16 = args.fp16
    scheduler_name = args.scheduler
    experiment = args.experiment
    dropout = args.dropout
    verbose = args.verbose
    accumulation_steps = args.accumulation_steps
    weight_decay = args.weight_decay
    balance = args.balance
    freeze_bn = args.freeze_bn
    train_batch_size = args.batch_size
    mixup = args.mixup
    cutmix = args.cutmix
    tsa = args.tsa
    obliterate_p = args.obliterate
    negative_image_dir = args.negative_image_dir

    # Compute batch size for validation
    valid_batch_size = train_batch_size

    current_time = datetime.now().strftime("%b%d_%H_%M")

    main_metric = "loss"
    main_metric_minimize = True

    for fold_index in range(5):
        x_train = np.load(f"embeddings_x_train_{fold_index}_Gf0_Gf3_Hnrmishf2_Hnrmishf1.npy")
        y_train = np.load(f"embeddings_y_train_{fold_index}_Gf0_Gf3_Hnrmishf2_Hnrmishf1.npy")

        x_valid = np.load(f"embeddings_x_valid_{fold_index}_Gf0_Gf3_Hnrmishf2_Hnrmishf1.npy")
        y_valid = np.load(f"embeddings_y_valid_{fold_index}_Gf0_Gf3_Hnrmishf2_Hnrmishf1.npy")

        train_ds = StackerDataset(x_train, y_train)
        valid_ds = StackerDataset(x_valid, y_valid)

        criterions_dict, loss_callbacks = get_criterions(
            modification_flag=modification_flag_loss,
            modification_type=None,
            embedding_loss=None,
            feature_maps_loss=None,
            mask_loss=None,
            bits_loss=None,
            num_epochs=num_epochs,
            mixup=mixup,
            cutmix=None,
            tsa=tsa,
        )

        callbacks = loss_callbacks + [
            OptimizerCallback(accumulation_steps=accumulation_steps, decouple_weight_decay=False),
            HyperParametersCallback(
                hparam_dict={
                    "scheduler": scheduler_name,
                    "optimizer": optimizer_name,
                    "augmentations": augmentations,
                    "weight_decay": weight_decay,
                }
            ),
        ]

        loaders = collections.OrderedDict()
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

        loaders["valid"] = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=num_workers, pin_memory=True)

        model: nn.Module = StackingModel(x_train.shape[1], dropout=0.25).cuda()

        optimizer = get_optimizer(
            optimizer_name, get_optimizable_parameters(model), learning_rate=learning_rate, weight_decay=weight_decay
        )
        scheduler = get_scheduler(
            scheduler_name, optimizer, lr=learning_rate, num_epochs=num_epochs, batches_in_epoch=len(loaders["train"])
        )
        if isinstance(scheduler, CyclicLR):
            callbacks += [SchedulerCallback(mode="batch")]

        checkpoint_prefix = f"{current_time}_stacking_fold{fold_index}"

        if fp16:
            checkpoint_prefix += "_fp16"

        if fast:
            checkpoint_prefix += "_fast"

        if mixup:
            checkpoint_prefix += "_mixup"

        if cutmix:
            checkpoint_prefix += "_cutmix"

        if experiment is not None:
            checkpoint_prefix = experiment

        log_dir = os.path.join("runs", checkpoint_prefix)
        os.makedirs(log_dir, exist_ok=False)

        config_fname = os.path.join(log_dir, f"{checkpoint_prefix}.json")
        with open(config_fname, "w") as f:
            train_session_args = vars(args)
            f.write(json.dumps(train_session_args, indent=2))

        print("Train session    :", checkpoint_prefix)
        print("  Train size     :", len(loaders["train"]), "batches", len(train_ds), "samples")
        print("  Valid size     :", len(loaders["valid"]), "batches", len(valid_ds), "samples")
        print("  FP16 mode      :", fp16)
        print("  Fast mode      :", args.fast)
        print("  Epochs         :", num_epochs)
        print("  Workers        :", num_workers)
        print("  Data dir       :", data_dir)
        print("  Log dir        :", log_dir)
        print("  Cache          :", cache)
        print("Data              ")
        print("  Augmentations  :", augmentations)
        print("  Obliterate (%) :", obliterate_p)
        print("  Negative images:", negative_image_dir)
        print("  Balance        :", balance)
        print("  Mixup          :", mixup)
        print("  CutMix         :", cutmix)
        print("  TSA            :", tsa)
        # print("Model            :", model_name)
        print("  Parameters     :", count_parameters(model))
        print("  Dropout        :", dropout)
        print("Optimizer        :", optimizer_name)
        print("  Learning rate  :", learning_rate)
        print("  Weight decay   :", weight_decay)
        print("  Scheduler      :", scheduler_name)
        print("  Batch sizes    :", train_batch_size, valid_batch_size)
        print("Losses            ")
        print("  Flag           :", modification_flag_loss)
        print("  Type           :", modification_type_loss)
        print("  Embedding      :", embedding_loss)
        print("  Feature maps   :", feature_maps_loss)
        print("  Mask           :", mask_loss)
        print("  Bits           :", bits_loss)

        # model training
        runner = SupervisedRunner(input_key=INPUT_EMBEDDING_KEY, output_key=None)
        runner.train(
            fp16=fp16,
            model=model,
            criterion=criterions_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loaders=loaders,
            logdir=os.path.join(log_dir, "main"),
            num_epochs=num_epochs,
            verbose=verbose,
            main_metric=main_metric,
            minimize_metric=main_metric_minimize,
            checkpoint_data={"cmd_args": vars(args)},
        )

        del optimizer, loaders, runner, callbacks

        best_checkpoint = os.path.join(log_dir, "main", "checkpoints", "best.pth")
        model_checkpoint = os.path.join(log_dir, f"{checkpoint_prefix}.pth")

        # Restore state of best model
        clean_checkpoint(best_checkpoint, model_checkpoint)
        # unpack_checkpoint(load_checkpoint(model_checkpoint), model=model)

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
