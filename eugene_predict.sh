# Environment variable KAGGLE_2020_ALASKA2 must be set prior running this script:
# export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

# Make holdout & test predictions that we will use later for stacking with XGB

python oof_predictions.py\
  models/G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/K_Jul17_17_09_nr_rgb_tf_efficientnet_b6_ns_mish_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/J_Jul19_20_10_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/H_Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/K_Jul18_16_41_nr_rgb_tf_efficientnet_b6_ns_mish_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  -d4\
  -b 32\
  -w 8

python make_submissions_xgb_gs.py\
  G_Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16\
  G_Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16\
  G_Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16\
  G_Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16\
  K_Jul17_17_09_nr_rgb_tf_efficientnet_b6_ns_mish_fold0_local_rank_0_fp16\
  J_Jul19_20_10_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16\
  H_Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16\
  K_Jul18_16_41_nr_rgb_tf_efficientnet_b6_ns_mish_fold3_local_rank_0_fp16\
  -o submits/xgb_cls_gs_Gf0_Gf1_Gf2_Gf3_Kmishf0_Jnrmishf1_Hnrmishf2_Kmishf3_with_logits.csv

# NOTE: The folllowing command needs ABBA and Eugene's prediction files to be computed

python blend.py\
  abba/submissions/submission_v26.csv\
  submits/xgb_cls_gs_Gf0_Gf1_Gf2_Gf3_Kmishf0_Jnrmishf1_Hnrmishf2_Kmishf3_with_logits.csv\
  -o submits/blend_10_ranked_v26_and_xgb_cls_gs_Gf0_Gf1_Gf2_Gf3_Kmishf0_Jnrmishf1_Hnrmishf2_Kmishf3_with_logits.csv