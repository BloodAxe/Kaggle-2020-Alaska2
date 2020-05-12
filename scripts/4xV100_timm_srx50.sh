export KAGGLE_2020_ALASKA2=/data/alaska2

python train2.py -m rgb_skresnext50_32x4d -b 56 -w 16 -d 0.2 -s flat_cos -o Ranger --epochs 50 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-3 --fold 2 --seed 2 --fp16 -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/May12_09_11_rgb_skresnext50_32x4d_fold2_fp16/main/checkpoints_auc/last.pth