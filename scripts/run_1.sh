export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
CUDA_VISIBLE_DEVICES=0 python train.py -m rgb_resnet34 -b 96 -w 16 -d 0.2 -s cos --epochs 50 --fine-tune 10 --show -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 0 --seed 12340 --fp16

export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
CUDA_VISIBLE_DEVICES=1 python train.py -m rgb_resnet34 -b 96 -w 16 -d 0.2 -s cos --epochs 50 --fine-tune 10 --show -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 1 --seed 12341 --fp16

export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
CUDA_VISIBLE_DEVICES=2 python train.py -m rgb_resnet34 -b 96 -w 16 -d 0.2 -s cos --epochs 50 --fine-tune 10 --show -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 2 --seed 12342 --fp16

export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
CUDA_VISIBLE_DEVICES=3 python train.py -m rgb_resnet34 -b 96 -w 16 -d 0.2 -s cos --epochs 50 --fine-tune 10 --show -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 3 --seed 12343 --fp16