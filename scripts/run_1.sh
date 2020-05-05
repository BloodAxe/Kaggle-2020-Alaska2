export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 0 --seed 12340

python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 1 --seed 12341

python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 2 --seed 12342

python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 3 --seed 12343
