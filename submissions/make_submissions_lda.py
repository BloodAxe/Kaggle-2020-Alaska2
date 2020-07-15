import os

# Used to ignore warnings generated from StackingCVClassifier
import warnings

# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.utils import fs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from alaska2 import get_holdout, INPUT_IMAGE_KEY, get_test_dataset
from alaska2.metric import alaska_weighted_auc
from alaska2.submissions import classifier_probas, sigmoid, parse_array
from submissions.eval_tta import get_predictions_csv
from submissions.make_submissions_averaging import compute_checksum_v2


def get_x_y(predictions):
    y = None
    X = []

    for p in predictions:
        p = pd.read_csv(p)
        if "true_modification_flag" in p:
            y = p["true_modification_flag"].values.astype(np.float32)

        # X.append(np.expand_dims(p["pred_modification_flag"].values, -1))
        # pred_modification_type = np.array(p["pred_modification_type"].apply(parse_array).tolist())
        # X.append(pred_modification_type)

        X.append(np.expand_dims(p["pred_modification_flag"].apply(sigmoid).values, -1))
        X.append(np.expand_dims(p["pred_modification_type"].apply(classifier_probas).values, -1))
        X.append(
            np.expand_dims(
                p["pred_modification_type"].apply(classifier_probas).values
                * p["pred_modification_flag"].apply(sigmoid).values,
                -1,
            )
        )

        if False and "pred_modification_type_tta" in p:
            def prase_tta_softmax(x):
                x = parse_array(x)
                x = torch.tensor(x)

                x = x.view((4,8))
                x = x.softmax(dim=0)

                # x = x.view((8,4))
                # x = x.softmax(dim=1)
                x = x.view(-1)
                return x.tolist()

            col = p["pred_modification_type_tta"].apply(prase_tta_softmax)
            X.append(col.tolist())

        if False and "pred_modification_flag_tta" in p:
            col = p["pred_modification_flag_tta"].apply(parse_array)
            col_act = col.apply(lambda x: torch.tensor(x).sigmoid().tolist())
            X.append(col_act.tolist())

        # if "pred_modification_type_tta" in p:
        #     X.append(p["pred_modification_type_tta"].apply(parse_array).tolist())
        #
        # if "pred_modification_flag_tta" in p:
        #     X.append(p["pred_modification_flag_tta"].apply(parse_array).tolist())

    X = np.column_stack(X).astype(np.float32)
    if y is not None:
        y = y.astype(int)
    return X, y


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
        "Jul12_18_42_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16",
    ]

    holdout_predictions = get_predictions_csv(experiments, "cauc", "holdout", "d4")
    test_predictions = get_predictions_csv(experiments, "cauc", "test", "d4")
    checksum = compute_checksum_v2(experiments)

    import torch.nn.functional as F

    holdout_ds = get_holdout("", features=[INPUT_IMAGE_KEY])
    image_ids = [fs.id_from_fname(x) for x in holdout_ds.images]

    quality_h = F.one_hot(torch.tensor(holdout_ds.quality).long(), 3).numpy().astype(np.float32)

    test_ds = get_test_dataset("", features=[INPUT_IMAGE_KEY])
    quality_t = F.one_hot(torch.tensor(test_ds.quality).long(), 3).numpy().astype(np.float32)

    x, y = get_x_y(holdout_predictions)
    print(x.shape, y.shape)

    x_test, _ = get_x_y(test_predictions)
    print(x_test.shape)

    if True:
        sc = StandardScaler()
        x = sc.fit_transform(x)
        x_test = sc.transform(x_test)

    if False:
        sc = PCA(n_components=16)
        x = sc.fit_transform(x)
        x_test = sc.transform(x_test)

    if True:
        x = np.column_stack([x, quality_h])
        x_test = np.column_stack([x_test, quality_t])

    group_kfold = GroupKFold(n_splits=5)
    cv_scores = []
    test_pred = None
    one_over_n = 1.0 / group_kfold.n_splits

    for train_index, valid_index in group_kfold.split(x, y, groups=image_ids):
        x_train, x_valid, y_train, y_valid = x[train_index], x[valid_index], y[train_index], y[valid_index]

        cls = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
        cls.fit(x_train, y_train)

        y_valid_pred = cls.predict_proba(x_valid)[:, 1]
        score = alaska_weighted_auc(y_valid, y_valid_pred)
        cv_scores.append(score)

        if test_pred is not None:
            test_pred += cls.predict_proba(x_test)[:, 1] * one_over_n
        else:
            test_pred = cls.predict_proba(x_test)[:, 1] * one_over_n

    for s in cv_scores:
        print(s)
    print(np.mean(cv_scores), np.std(cv_scores))

    submit_fname = os.path.join(output_dir, f"lda_{np.mean(cv_scores):.4f}_{checksum}.csv")
    df = pd.read_csv(test_predictions[0]).rename(columns={"image_id": "Id"})
    df["Label"] = test_pred
    df[["Id", "Label"]].to_csv(submit_fname, index=False)
    print("Saved submission to ", submit_fname)


if __name__ == "__main__":
    main()
