import warnings

from alaska2.submissions import parse_classifier_probas, sigmoid

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from collections import defaultdict
from catalyst.utils import any2device
from pytorch_toolbelt.utils import to_numpy, fs
from pytorch_toolbelt.utils.catalyst import report_checkpoint

import argparse
import os
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from alaska2 import *
from predict import compute_test_predictions


@torch.no_grad()
def compute_oof_predictions(model, dataset, batch_size=1, workers=0) -> pd.DataFrame:
    df = defaultdict(list)
    for batch in tqdm(
        DataLoader(
            dataset, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=False, pin_memory=True
        )
    ):
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
    parser.add_argument("-oof", "--need-oof", action="store_true")

    args = parser.parse_args()

    checkpoint_fnames = args.checkpoint
    data_dir = args.data_dir
    batch_size = args.batch_size
    workers = args.workers

    d4_tta = args.d4_tta
    hv_tta = args.hv_tta
    force_recompute = args.force_recompute
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

        # Holdout
        variants = {
            "istego100k_test_same_center_crop": get_istego100k_test_same(
                data_dir, features=required_features, output_size="center_crop"
            ),
            "istego100k_test_same_full": get_istego100k_test_same(
                data_dir, features=required_features, output_size="full"
            ),
            "istego100k_test_other_center_crop": get_istego100k_test_other(
                data_dir, features=required_features, output_size="center_crop"
            ),
            "istego100k_test_other_full": get_istego100k_test_other(
                data_dir, features=required_features, output_size="full"
            ),
            "holdout": get_holdout("d:\datasets\ALASKA2", features=required_features),
        }

        for name, dataset in variants.items():
            print("Making predictions for ", name, len(dataset))

            predictions_csv = fs.change_extension(checkpoint_fname, f"_{name}_predictions.csv")
            if force_recompute or not os.path.exists(predictions_csv):
                holdout_predictions = compute_oof_predictions(
                    model, dataset, batch_size=batch_size // 4 if "full" in name else batch_size, workers=workers
                )
                holdout_predictions.to_csv(predictions_csv, index=False)
                holdout_predictions = pd.read_csv(predictions_csv)

                print(name)
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


if __name__ == "__main__":
    main()
