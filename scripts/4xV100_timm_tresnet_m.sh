export KAGGLE_2020_ALASKA2=/data/alaska2

python train.py -m rgb_tresnet_m -b 256 -w 16 -d 0.2 -s flat_cos -o Ranger --epochs 100 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-3 --fold 0 --seed 10 --fp16

python train.py -m rgb_tresnet_m -b 256 -w 16 -d 0.2 -s flat_cos -o Ranger --epochs 100 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-3 --fold 1 --seed 11 --fp16

python train.py -m rgb_tresnet_m -b 256 -w 16 -d 0.2 -s flat_cos -o Ranger --epochs 100 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-3 --fold 2 --seed 12 --fp16

python train.py -m rgb_tresnet_m -b 256 -w 16 -d 0.2 -s flat_cos -o Ranger --epochs 100 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -lr 1e-3 --fold 3 --seed 13 --fp16