import os

# For reading, visualizing, and preprocessing data
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_toolbelt.utils import fs
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.submissions import get_x_y_embedding_for_stacking
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2


# Used to ignore warnings generated from StackingCVClassifier


def main():
    output_dir = os.path.dirname(__file__)

    experiments = [
        # "A_May24_11_08_ela_skresnext50_32x4d_fold0_fp16",
        # "A_May15_17_03_ela_skresnext50_32x4d_fold1_fp16",
        # "A_May21_13_28_ela_skresnext50_32x4d_fold2_fp16",
        # "A_May26_12_58_ela_skresnext50_32x4d_fold3_fp16",
        #
        # "B_Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "B_Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        # "B_Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        # "B_Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        # "C_Jun24_22_00_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16",
        #
        # "D_Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16",
        # "D_Jun20_09_52_rgb_tf_efficientnet_b7_ns_fold2_local_rank_0_fp16",
        #
        # "E_Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "E_Jun21_10_48_rgb_tf_efficientnet_b6_ns_fold0_istego100k_local_rank_0_fp16",
        #
        # "F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16",
        #
        "G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16",
        # "G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16",
        # "G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16",
        "G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16",
        #
        "H_Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16",
        "H_Jul12_18_42_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
    ]

    # holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    # test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", tta=None, need_embedding=True)
    test_predictions = get_predictions_csv(experiments, "cauc", "test", tta=None, need_embedding=True)
    checksum = compute_checksum_v2(experiments)

    holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
    image_ids = [fs.id_from_fname(x) for x in holdout_ds.images]

    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    x, y = get_x_y_embedding_for_stacking(holdout_predictions)
    print(x.shape, y.shape)

    x_test, _ = get_x_y_embedding_for_stacking(test_predictions)
    print(x_test.shape)

    if False:
        sc = StandardScaler()
        x = sc.fit_transform(x)
        x_test = sc.transform(x_test)

    if True:
        x = np.column_stack([x, quality_h])
        x_test = np.column_stack([x_test, quality_t])

    group_kfold = GroupKFold(n_splits=5)

    np.save(f"embeddings_x_test_{checksum}.npy", x_test)

    for fold_index, (train_index, valid_index) in enumerate(group_kfold.split(x, y, groups=image_ids)):
        x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]

        np.save(f"embeddings_x_train_{fold_index}_{checksum}.npy", x_train)
        np.save(f"embeddings_x_valid_{fold_index}_{checksum}.npy", x_valid)

        np.save(f"embeddings_y_train_{fold_index}_{checksum}.npy", y_train)
        np.save(f"embeddings_y_valid_{fold_index}_{checksum}.npy", y_valid)


if __name__ == "__main__":
    main()
