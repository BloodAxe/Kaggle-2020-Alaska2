export KAGGLE_2020_ALASKA2=/data/alaska2

#python train.py -m rgb_skresnext50_32x4d -b 112 -w 16 -d 0.2 -s flat_cos -o Ranger --epochs 50 -a light\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-3 --fold 2 --seed 2 --fp16

python train.py -m rgb_skresnext50_32x4d -b 112 -w 16 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-3 --fold 1 --seed 1 --fp16