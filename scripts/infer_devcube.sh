export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python oof_predictions.py\
  models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best.pth\
  models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best.pth\
  \
  models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/best.pth\
  models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best.pth\
  \
  models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc/best.pth\
  models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best.pth\
  \
  models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth\
  models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc/best.pth\
  models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best.pth\
  \
  -b 64 -w 4 -f -d4 -hv
