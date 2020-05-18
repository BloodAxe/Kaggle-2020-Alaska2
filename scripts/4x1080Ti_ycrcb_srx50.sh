export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train.py -m ycrcb_skresnext50_32x4d -b 284 -v -w 24 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-2 --fold 0 --seed 0 --fp16
