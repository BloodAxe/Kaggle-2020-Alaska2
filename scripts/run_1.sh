export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

#python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 0 --seed 12340
#
#python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 1 --seed 12341
#
#python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 2 --seed 12342
#
#python train.py -m rgb_dct_resnet34 -b 160 -w 16 -d 0.2 -s cos --warmup 50 --epochs 50 --fine-tune 50 --show -a hard\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 3 --seed 12343

#python train.py -m rgb_resnet34 -b 160 -w 16 -d 0.2 -s cos --epochs 50 --fine-tune 50 --show -a light\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 0 --seed 123400 --transfer runs/May05_22_29_rgb_dct_resnet34_fold0/May05_22_29_rgb_dct_resnet34_fold0_warmup.pth

#python train.py -m dct_resnet34 -b 512 -w 12 -d 0.2 -s cos --epochs 50 --fine-tune 0 --show -a light\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 0 --seed 123400 --transfer runs/May06_14_43_dct_resnet34_fold0/main/checkpoints_auc/best.pth

python train.py -m ela_resnet34 -b 64 -w 8 -d 0.2 -s cos --epochs 50 --fine-tune 0 --show -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-3 --fold 0 --seed 123400