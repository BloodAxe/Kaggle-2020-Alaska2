export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train.py -m ela_rich_skresnext50_32x4d -b 72 -w 8 -d 0.2 -s flat_cos -o Ranger --epochs 25 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1\
  -lr 1e-3 --fold 1 --seed 111 --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best.pth


