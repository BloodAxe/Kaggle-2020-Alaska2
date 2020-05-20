export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train4.py -m ela_skresnext50_32x4d -b 19 -w 24 -d 0.2 -s flat_cos -o Ranger --epochs 25 -a light\
  --modification-flag-loss rank2 1 -lr 1e-4 --fold 1 --seed 1 --fp16 -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/models/May15_17_03_ela_skresnext50_32x4d_fold1_fp16/best.pth
