export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train.py -m hpf_net_v2 -b 96 -w 24 -d 0.2 -s flat_cos -o Ranger --epochs 25 -a light\
  --modification-flag-loss rank2 1\
  --modification-type-loss ce 1\
  -lr 1e-4 --fold 1 --seed 111 --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/May21_10_14_hpf_net_fold0_fp16/main/checkpoints_auc_classifier/best.pth
