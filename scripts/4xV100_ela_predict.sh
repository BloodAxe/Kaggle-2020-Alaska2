export KAGGLE_2020_ALASKA2=/data/alaska2

#python predict.py\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc_classifier/best.pth\
#  \
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc_classifier/best.pth\
#  \
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc_classifier/best.pth\
#  \
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc/best.pth\
#  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc_classifier/best.pth\
#  \
#  -b 256\
#  -w 4


python oof_predictions.py\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/main/checkpoints_auc_classifier/best.pth\
  \
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May21_13_28_ela_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc_classifier/best.pth\
  \
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May24_11_08_ela_skresnext50_32x4d_fold0_fp16/main/checkpoints_auc_classifier/best.pth\
  \
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc/best.pth\
  /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May26_12_58_ela_skresnext50_32x4d_fold3_fp16/main/checkpoints_auc_classifier/best.pth\
  \
  -b 256\
  -w 4

