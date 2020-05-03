export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
python train.py -m rgb_dct_resnet34 -b 128 -w 16 -d 0.2 -s cos --warmup 50 --epochs 100 --fine-tune 50 --show -a hard\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v