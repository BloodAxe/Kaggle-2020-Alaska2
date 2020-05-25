import pandas as pd
import numpy as np
import torch
from pytorch_toolbelt.utils import to_numpy, logit

from alaska2 import alaska_weighted_auc
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as IR


def test_calibartion():
    oof_predictions = pd.read_csv(
        "/old_models/May07_16_48_rgb_resnet34_fold0/oof_predictions.csv"
    )

    print("Uncalibrated", alaska_weighted_auc(oof_predictions["y_true"].values, oof_predictions["y_pred"].values))

    # ir = IR(out_of_bounds="clip")
    # ir.fit(oof_predictions["y_pred"].values, oof_predictions["y_true"].values)
    # p_calibrated = ir.transform(oof_predictions["y_pred"].values)
    # print("IR", alaska_weighted_auc(oof_predictions["y_true"].values, p_calibrated))
    #
    # lr = LR()
    # lr.fit(oof_predictions["y_pred"].values.reshape(-1, 1), oof_predictions["y_true"].values)
    # p_calibrated = lr.predict_proba(oof_predictions["y_pred"].values.reshape(-1, 1))
    # print("LR", alaska_weighted_auc(oof_predictions["y_true"].values, p_calibrated[:, 1]))

    x = torch.from_numpy(oof_predictions["y_pred"].values)
    x = torch.sigmoid(logit(x) * 100)
    x = to_numpy(x)

    print("Temp", alaska_weighted_auc(oof_predictions["y_true"].values, x))

    # with open(os.path.join(output_dir, "calibration.pkl"), "wb") as f:
    #     pickle.dump(ir, f)

    # loader = DataLoader(
    #     test_ds, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=False, drop_last=False
    # )
