import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from collections import defaultdict
from catalyst.utils import any2device
from pytorch_toolbelt.utils import to_numpy, fs
from pytorch_toolbelt.utils.catalyst import report_checkpoint

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import argparse
import os

import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from alaska2 import *


@torch.no_grad()
def compute_oof_predictions(model, dataset, batch_size=1, workers=0) -> pd.DataFrame:

    df = defaultdict(list)
    for batch in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)):
        batch = any2device(batch, device="cuda")

        image_ids = batch[INPUT_IMAGE_ID_KEY]
        y_trues = to_numpy(batch[INPUT_TRUE_MODIFICATION_FLAG]).flatten()
        y_labels = to_numpy(batch[INPUT_TRUE_MODIFICATION_TYPE]).flatten()

        df[INPUT_IMAGE_ID_KEY].extend(image_ids)
        df[INPUT_TRUE_MODIFICATION_FLAG].extend(y_trues)
        df[INPUT_TRUE_MODIFICATION_TYPE].extend(y_labels)

        outputs = model(**batch)
        if OUTPUT_PRED_MODIFICATION_FLAG in outputs:
            df[OUTPUT_PRED_MODIFICATION_FLAG].extend(to_numpy(outputs[OUTPUT_PRED_MODIFICATION_FLAG]).flatten())

        if OUTPUT_PRED_MODIFICATION_TYPE in outputs:
            df[OUTPUT_PRED_MODIFICATION_TYPE].extend(to_numpy(outputs[OUTPUT_PRED_MODIFICATION_TYPE]).tolist())

    df = pd.DataFrame.from_dict(df)
    return df


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

    args = parser.parse_args()

    checkpoint_fnames = args.checkpoint
    data_dir = args.data_dir
    batch_size = args.batch_size
    workers = args.workers

    d4_tta = args.d4_tta
    hv_tta = args.hv_tta

    outputs = [OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_MODIFICATION_TYPE]

    for checkpoint_fname in checkpoint_fnames:

        model, checkpoints, required_features = ensemble_from_checkpoints(
            [checkpoint_fname], strict=True, outputs=outputs, activation=None, tta=None
        )

        report_checkpoint(checkpoints[0])

        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.eval()

        fold = checkpoints[0]["checkpoint_data"]["cmd_args"]["fold"]
        _, valid_ds, _ = get_datasets(data_dir, fold=fold, features=required_features)

        oof_predictions = compute_oof_predictions(model, valid_ds, batch_size=batch_size, workers=workers)
        oof_predictions_csv = fs.change_extension(checkpoint_fname, "_oof_predictions.csv")
        oof_predictions.to_csv(oof_predictions_csv, index=False)

        if hv_tta:
            tta_model = wrap_model_with_tta(model, "flip-hv", inputs=required_features, outputs=outputs).eval()
            oof_predictions = compute_oof_predictions(tta_model, valid_ds, batch_size=batch_size, workers=workers)
            oof_predictions_csv = fs.change_extension(checkpoint_fname, "_oof_predictions_flip_hv_tta.csv")
            oof_predictions.to_csv(oof_predictions_csv, index=False)

        if d4_tta:
            tta_model = wrap_model_with_tta(model, "d4", inputs=required_features, outputs=outputs).eval()
            oof_predictions = compute_oof_predictions(tta_model, valid_ds, batch_size=batch_size, workers=workers)
            oof_predictions_csv = fs.change_extension(checkpoint_fname, "_oof_predictions_d4_tta.csv")
            oof_predictions.to_csv(oof_predictions_csv, index=False)

        # Holdout
        holdout_ds = get_holdout(data_dir, features=required_features)

        holdout_predictions = compute_oof_predictions(model, holdout_ds, batch_size=batch_size, workers=workers)
        holdout_predictions_csv = fs.change_extension(checkpoint_fname, "_holdout_predictions.csv")
        holdout_predictions.to_csv(holdout_predictions_csv, index=False)

        if hv_tta:
            tta_model = wrap_model_with_tta(model, "flip-hv", inputs=required_features, outputs=outputs).eval()
            holdout_predictions = compute_oof_predictions(
                tta_model, holdout_ds, batch_size=batch_size, workers=workers
            )
            holdout_predictions_csv = fs.change_extension(checkpoint_fname, "_holdout_predictions_flip_hv_tta.csv")
            holdout_predictions.to_csv(holdout_predictions_csv, index=False)

        if d4_tta:
            tta_model = wrap_model_with_tta(model, "d4", inputs=required_features, outputs=outputs).eval()
            holdout_predictions = compute_oof_predictions(
                tta_model, holdout_ds, batch_size=batch_size, workers=workers
            )
            holdout_predictions_csv = fs.change_extension(checkpoint_fname, "_holdout_predictions_d4_tta.csv")
            holdout_predictions.to_csv(holdout_predictions_csv, index=False)


if __name__ == "__main__":
    main()
