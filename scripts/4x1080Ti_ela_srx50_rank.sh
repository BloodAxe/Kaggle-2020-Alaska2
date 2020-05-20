export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train2.py -m ela_skresnext50_32x4d -b 38 -w 24 -d 0.2 -s flat_cos -o Ranger --epochs 10 -a light\
  --modification-flag-loss rank2 1 -lr 1e-3 --fold 0 --seed 0 --fp16

python train2.py -m ela_skresnext50_32x4d -b 38 -w 24 -d 0.2 -s flat_cos -o Ranger --epochs 10 -a light\
  --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 --fp16
