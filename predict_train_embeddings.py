import warnings


warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import argparse
import os
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from collections import defaultdict
from catalyst.utils import any2device
from pytorch_toolbelt.utils import to_numpy, fs
from pytorch_toolbelt.utils.catalyst import report_checkpoint

from alaska2 import *
from alaska2.dataset import get_train_except_holdout


@torch.no_grad()
def compute_trn_predictions(model, dataset, batch_size=1, workers=0) -> pd.DataFrame:
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
            df[OUTPUT_PRED_MODIFICATION_TYPE].extend(outputs[OUTPUT_PRED_MODIFICATION_TYPE].tolist())

        if OUTPUT_PRED_EMBEDDING in outputs:
            df[OUTPUT_PRED_EMBEDDING].extend(outputs[OUTPUT_PRED_EMBEDDING].tolist())

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

    args = parser.parse_args()

    checkpoint_fnames = args.checkpoint
    data_dir = args.data_dir
    batch_size = args.batch_size
    workers = args.workers

    d4_tta = args.d4_tta
    force_recompute = args.force_recompute
    need_embedding = True

    outputs = [OUTPUT_PRED_MODIFICATION_FLAG, OUTPUT_PRED_MODIFICATION_TYPE, OUTPUT_PRED_EMBEDDING]
    embedding_suffix = "_w_emb" if need_embedding else ""

    for checkpoint_fname in checkpoint_fnames:
        model, checkpoints, required_features = ensemble_from_checkpoints(
            [checkpoint_fname], strict=True, outputs=outputs, activation=None, tta=None, need_embedding=need_embedding
        )

        report_checkpoint(checkpoints[0])

        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.eval()

        train_ds = get_train_except_holdout(data_dir, features=required_features)
        holdout_ds = get_holdout(data_dir, features=required_features)
        test_ds = get_test_dataset(data_dir, features=required_features)

        if d4_tta:
            model = wrap_model_with_tta(model, "d4", inputs=required_features, outputs=outputs).eval()
            tta_suffix = "_d4_tta"
        else:
            tta_suffix = ""

        # Train
        trn_predictions_csv = fs.change_extension(
            checkpoint_fname, f"_train_predictions{embedding_suffix}{tta_suffix}.pkl"
        )
        if force_recompute or not os.path.exists(trn_predictions_csv):
            trn_predictions = compute_trn_predictions(model, train_ds, batch_size=batch_size, workers=workers)
            trn_predictions.to_pickle(trn_predictions_csv)

        # Holdout
        hld_predictions_csv = fs.change_extension(
            checkpoint_fname, f"_holdout_predictions{embedding_suffix}{tta_suffix}.pkl"
        )
        if force_recompute or not os.path.exists(hld_predictions_csv):
            hld_predictions = compute_trn_predictions(model, holdout_ds, batch_size=batch_size, workers=workers)
            hld_predictions.to_pickle(hld_predictions_csv)

        # Test
        tst_predictions_csv = fs.change_extension(
            checkpoint_fname, f"_test_predictions{embedding_suffix}{tta_suffix}.pkl"
        )
        if force_recompute or not os.path.exists(tst_predictions_csv):
            tst_predictions = compute_trn_predictions(model, test_ds, batch_size=batch_size, workers=workers)
            tst_predictions.to_pickle(tst_predictions_csv)


if __name__ == "__main__":
    main()
