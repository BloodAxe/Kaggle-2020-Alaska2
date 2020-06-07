import os

from submissions.ela_skresnext50_32x4d import *
from submissions.rgb_tf_efficientnet_b2_ns import *
from submissions.rgb_tf_efficientnet_b6_ns import *
from alaska2.submissions import (
    submit_from_classifier_calibrated,
    submit_from_average_classifier,
    blend_predictions_ranked,
    make_classifier_predictions,
    make_classifier_predictions_calibrated,
    make_binary_predictions_calibrated,
    blend_predictions_mean,
    as_hv_tta,
    as_d4_tta,
)


def main():
    output_dir = os.path.dirname(__file__)

    if False:
        # 0.917
        submit_from_average_classifier([rgb_tf_efficientnet_b6_ns_best_auc_c[0]]).to_csv(
            os.path.join(output_dir, "May28_13_04_rgb_tf_efficientnet_b6_ns_fold0.csv"), index=False
        )

    if False:
        # 0.915
        submit_from_classifier_calibrated(
            [rgb_tf_efficientnet_b6_ns_best_auc_c[0]], [rgb_tf_efficientnet_b6_ns_best_auc_c_oof[0]]
        ).to_csv(os.path.join(output_dir, "May28_13_04_rgb_tf_efficientnet_b6_ns_fold0_calibrated.csv"), index=False)

    # 0.930
    if False:
        # Take best classifier checkpoints for all models and average them using ranking
        predictions = make_classifier_predictions(
            ela_skresnext50_32x4d_best_auc_c
            + rgb_tf_efficientnet_b6_ns_best_auc_c
            + rgb_tf_efficientnet_b2_ns_best_auc_c
        )
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "1xb2_2xb6_4xskrx50_classifier_ranked.csv"), index=False
        )

    # 0.930
    if False:
        # Take best classifier checkpoints for all models and average them using ranking
        # HV TTA
        predictions = make_classifier_predictions(
            as_hv_tta(
                ela_skresnext50_32x4d_best_auc_c
                + rgb_tf_efficientnet_b6_ns_best_auc_c
                + rgb_tf_efficientnet_b2_ns_best_auc_c
            )
        )
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "1xb2_2xb6_4xskrx50_hv_tta_classifier_ranked.csv"), index=False
        )

    # 0.929
    if False:
        # Take best classifier checkpoints for all models and average them using ranking
        # D4 TTA
        predictions = make_classifier_predictions(
            as_d4_tta(
                ela_skresnext50_32x4d_best_auc_c
                + rgb_tf_efficientnet_b6_ns_best_auc_c
                + rgb_tf_efficientnet_b2_ns_best_auc_c
            )
        )
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "1xb2_2xb6_4xskrx50_d4_tta_classifier_ranked.csv"), index=False
        )

    # 0.931
    if False:
        # Take best classifier checkpoints for all models and average them using ranking
        predictions = make_classifier_predictions(
            ela_skresnext50_32x4d_best_loss
            + rgb_tf_efficientnet_b6_ns_best_loss
            + rgb_tf_efficientnet_b2_ns_best_auc_c
        )
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "1xb2_2xb6_4xskrx50_best_loss_classifier_ranked.csv"), index=False
        )

    # 0.930
    if False:
        # Take best classifier checkpoints for all models and average them using ranking
        predictions = make_classifier_predictions_calibrated(
            ela_skresnext50_32x4d_best_loss
            + rgb_tf_efficientnet_b6_ns_best_loss
            + rgb_tf_efficientnet_b2_ns_best_auc_c,
            ela_skresnext50_32x4d_best_loss_oof
            + rgb_tf_efficientnet_b6_ns_best_loss_oof
            + rgb_tf_efficientnet_b2_ns_best_auc_c_oof,
        )
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "1xb2_2xb6_4xskrx50_best_loss_classifier_calibrated_ranked.csv"), index=False
        )

    # 0.930
    if False:
        # Take best classifier checkpoints for all models and average them using ranking
        predictions = make_classifier_predictions_calibrated(
            ela_skresnext50_32x4d_best_loss + rgb_tf_efficientnet_b6_ns_best_loss,
            # + rgb_tf_efficientnet_b2_ns_best_auc_c,
            ela_skresnext50_32x4d_best_loss_oof + rgb_tf_efficientnet_b6_ns_best_loss_oof
            # + rgb_tf_efficientnet_b2_ns_best_auc_c_oof,
        )
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "2xb6_4xskrx50_best_loss_classifier_calibrated_ranked.csv"), index=False
        )

    # 0.930
    if False:
        # Take best classifier checkpoints for all models and average them using ranking
        predictions = make_classifier_predictions_calibrated(
            ela_skresnext50_32x4d_best_loss
            + ela_skresnext50_32x4d_best_auc_b
            + ela_skresnext50_32x4d_best_auc_c
            + rgb_tf_efficientnet_b6_ns_best_loss
            + rgb_tf_efficientnet_b6_ns_best_auc_b
            + rgb_tf_efficientnet_b6_ns_best_auc_c
            + rgb_tf_efficientnet_b2_ns_best_auc_c,
            ela_skresnext50_32x4d_best_loss_oof
            + ela_skresnext50_32x4d_best_auc_b_oof
            + ela_skresnext50_32x4d_best_auc_c_oof
            + rgb_tf_efficientnet_b6_ns_best_loss_oof
            + rgb_tf_efficientnet_b6_ns_best_auc_b_oof
            + rgb_tf_efficientnet_b6_ns_best_auc_c_oof
            + rgb_tf_efficientnet_b2_ns_best_auc_c_oof,
        )
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "1xb2_2xb6_4xskrx50_best_lcb_classifier_calibrated_ranked.csv"), index=False
        )

    if False:
        # Fold 0
        predictions = (
            make_classifier_predictions([rgb_tf_efficientnet_b6_ns_best_auc_b[0]])
            + make_classifier_predictions_calibrated(
                as_d4_tta([ela_skresnext50_32x4d_best_loss[1]]), as_d4_tta([ela_skresnext50_32x4d_best_loss_oof[1]])
            )
            + make_binary_predictions_calibrated(
                [ela_skresnext50_32x4d_best_loss[2]], [ela_skresnext50_32x4d_best_loss_oof[2]]
            )
            + make_classifier_predictions_calibrated(
                [ela_skresnext50_32x4d_best_auc_b[3]], [ela_skresnext50_32x4d_best_auc_b_oof[3]]
            )
        )

        # 0.931
        blend_predictions_ranked(predictions).to_csv(
            os.path.join(output_dir, "best_models_for_each_fold_ranked.csv"), index=False
        )

        # 0.929
        blend_predictions_mean(predictions).to_csv(
            os.path.join(output_dir, "best_models_for_each_fold_mean.csv"), index=False
        )

    if True:
        p1 = blend_predictions_mean(make_classifier_predictions(rgb_tf_efficientnet_b6_ns_best_auc_c))
        p2 = blend_predictions_mean(make_classifier_predictions(rgb_tf_efficientnet_b6_ns_best_loss))
        p3 = blend_predictions_mean(make_classifier_predictions(ela_skresnext50_32x4d_best_loss))
        p4 = blend_predictions_mean(make_classifier_predictions(rgb_tf_efficientnet_b2_ns_best_auc_c))

        blend_predictions_ranked([p1,p2,p3,p4]).to_csv(
            os.path.join(output_dir, "averaged_folds_ensemble_ranked.csv"), index=False
        )


if __name__ == "__main__":
    main()
