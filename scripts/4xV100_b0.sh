export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train.py -m rgb_b0 -b 184 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 3e-4 --fold 0 --seed 12350

python train.py -m rgb_b0 -b 184 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 3e-4 --fold 1 --seed 12351

python train.py -m rgb_b0 -b 184 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 3e-4 --fold 2 --seed 12352

python train.py -m rgb_b0 -b 184 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 3e-4 --fold 3 --seed 12353
