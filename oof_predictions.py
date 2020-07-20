import warnings

from alaska2.adabn import bn_update

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import argparse
import os
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from collections import defaultdict
from catalyst.utils import any2device
from pytorch_toolbelt.utils import to_numpy, fs
from pytorch_toolbelt.utils.catalyst import report_checkpoint

from alaska2 import *
from alaska2.submissions import sigmoid, parse_classifier_probas


def update_bn(model: nn.Module, dataset: Dataset, batch_size=1, workers=0):
    """
    BatchNorm buffers update (if any).
    Performs 1 epochs to estimate buffers average using train dataset.
    :param loader: train dataset loader for buffers average estimation.
    :param model: model being update
    :return: None
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True, pin_memory=True)
    bn_update(loader, model)


@torch.no_grad()
def compute_oof_predictions(model, dataset: Dataset, batch_size=1, workers=0) -> pd.DataFrame:
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.eval()

    df = defaultdict(list)
    for batch in tqdm(
        DataLoader(
            dataset, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=False, pin_memory=True
        )
    ):
        batch = any2device(batch, device="cuda")

        if INPUT_TRUE_MODIFICATION_FLAG in batch:
            y_trues = to_numpy(batch[INPUT_TRUE_MODIFICATION_FLAG]).flatten()
            df[INPUT_TRUE_MODIFICATION_FLAG].extend(y_trues)

        if INPUT_TRUE_MODIFICATION_TYPE in batch:
            y_labels = to_numpy(batch[INPUT_TRUE_MODIFICATION_TYPE]).flatten()
            df[INPUT_TRUE_MODIFICATION_TYPE].extend(y_labels)

        image_ids = batch[INPUT_IMAGE_ID_KEY]
        df[INPUT_IMAGE_ID_KEY].extend(image_ids)

        outputs = model(**batch)

        if OUTPUT_PRED_MODIFICATION_FLAG in outputs:
            df[OUTPUT_PRED_MODIFICATION_FLAG].extend(to_numpy(outputs[OUTPUT_PRED_MODIFICATION_FLAG]).flatten())

        if OUTPUT_PRED_MODIFICATION_TYPE in outputs:
            df[OUTPUT_PRED_MODIFICATION_TYPE].extend(to_numpy(outputs[OUTPUT_PRED_MODIFICATION_TYPE]).tolist())

        if OUTPUT_PRED_EMBEDDING in outputs:
            df[OUTPUT_PRED_EMBEDDING].extend(to_numpy(outputs[OUTPUT_PRED_EMBEDDING]).tolist())

        if OUTPUT_PRED_EMBEDDING_ARC_MARGIN in outputs:
            df[OUTPUT_PRED_EMBEDDING_ARC_MARGIN].extend(to_numpy(outputs[OUTPUT_PRED_EMBEDDING_ARC_MARGIN]).tolist())

        # Save also TTA predictions for future use
        if OUTPUT_PRED_MODIFICATION_FLAG + "_tta" in outputs:
            df[OUTPUT_PRED_MODIFICATION_FLAG + "_tta"].extend(
                to_numpy(outputs[OUTPUT_PRED_MODIFICATION_FLAG + "_tta"]).tolist()
            )

        if OUTPUT_PRED_MODIFICATION_TYPE + "_tta" in outputs:
            df[OUTPUT_PRED_MODIFICATION_TYPE + "_tta"].extend(
                to_numpy(outputs[OUTPUT_PRED_MODIFICATION_TYPE + "_tta"]).tolist()
            )

    df = pd.DataFrame.from_dict(df)
    return df


def score_predictions(predictions_fname):
    holdout_predictions = pd.read_csv(predictions_fname)

    print(predictions_fname)
    print(
        "\tbAUC",
        alaska_weighted_auc(
            holdout_predictions[INPUT_TRUE_MODIFICATION_FLAG].values,
            holdout_predictions[OUTPUT_PRED_MODIFICATION_FLAG].apply(sigmoid).values,
        ),
    )

    print(
        "\tcAUC",
        alaska_weighted_auc(
            holdout_predictions[INPUT_TRUE_MODIFICATION_FLAG].values,
            holdout_predictions[OUTPUT_PRED_MODIFICATION_TYPE].apply(parse_classifier_probas).values,
        ),
    )


@torch.no_grad()
def main():
    # Give no chance to randomness
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="+")
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-w", "--workers", type=int, default=0)
    parser.add_argument("-d4", "--d4-tta", action="store_true")
    parser.add_argument("-hv", "--hv-tta", action="store_true")
    parser.add_argument("-f", "--force-recompute", action="store_true")
    parser.add_argument("-oof", "--need-oof", action="store_true")
    parser.add_argument("-emb", "--need-embedding", action="store_true")
    parser.add_argument("-adabn", "--adabn", action="store_true")

    args = parser.parse_args()

    checkpoint_fnames = args.checkpoint
    data_dir = args.data_dir
    batch_size = args.batch_size
    workers = args.workers

    d4_tta = args.d4_tta
    hv_tta = args.hv_tta
    force_recompute = args.force_recompute
    need_embedding = args.need_embedding
    adabn = args.adabn

    outputs = [OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_MODIFICATION_TYPE]
    suffix = (
        ("_w_emb" if need_embedding else "")
        + ("_adabn" if adabn else "")
        + ("_flip_hv_tta" if hv_tta else "")
        + ("_d4_tta" if d4_tta else "")
    )

    for checkpoint_fname in checkpoint_fnames:
        model, checkpoints, required_features = ensemble_from_checkpoints(
            [checkpoint_fname], strict=True, outputs=outputs, activation=None, tta=None, need_embedding=need_embedding
        )

        report_checkpoint(checkpoints[0])
        model = model.cuda()
        if hv_tta:
            model = wrap_model_with_tta(model, "flip-hv", inputs=required_features, outputs=outputs).eval()
        elif d4_tta:
            model = wrap_model_with_tta(model, "d4", inputs=required_features, outputs=outputs).eval()

        if args.need_oof:
            fold = checkpoints[0]["checkpoint_data"]["cmd_args"]["fold"]
            _, valid_ds, _ = get_datasets(data_dir, fold=fold, features=required_features)

            oof_predictions_csv = fs.change_extension(checkpoint_fname, f"_oof_predictions{suffix}.csv")
            if force_recompute or not os.path.exists(oof_predictions_csv):
                oof_predictions = compute_oof_predictions(model, valid_ds, batch_size=batch_size, workers=workers)
                oof_predictions.to_csv(oof_predictions_csv, index=False)
            print(f"OOF score ({suffix})")
            score_predictions(oof_predictions_csv)

        # Holdout
        holdout_ds = get_holdout(data_dir, features=required_features)
        holdout_predictions_csv = fs.change_extension(checkpoint_fname, f"_holdout_predictions{suffix}.csv")
        if force_recompute or not os.path.exists(holdout_predictions_csv):
            if adabn:
                update_bn(model, holdout_ds, batch_size=batch_size // torch.cuda.device_count(), workers=workers)
            holdout_predictions = compute_oof_predictions(model, holdout_ds, batch_size=batch_size, workers=workers)
            holdout_predictions.to_csv(holdout_predictions_csv, index=False)
        print(f"Holdout score ({suffix})")
        score_predictions(holdout_predictions_csv)

        # Test
        test_ds = get_test_dataset(data_dir, features=required_features)
        test_predictions_csv = fs.change_extension(checkpoint_fname, f"_test_predictions{suffix}.csv")
        if force_recompute or not os.path.exists(test_predictions_csv):
            if adabn:
                update_bn(model, test_ds, batch_size=batch_size // torch.cuda.device_count(), workers=workers)
            test_predictions = compute_oof_predictions(model, test_ds, batch_size=batch_size, workers=workers)
            test_predictions.to_csv(test_predictions_csv, index=False)


if __name__ == "__main__":
    main()
