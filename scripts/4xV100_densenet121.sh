export KAGGLE_2020_ALASKA2=/data/alaska2

#python train.py -m rgb_densenet121 -b 160 -w 24 -s cos --epochs 100 --fine-tune 15 -a safe\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.01 -v -lr 1e-4 --fold 0 --seed 0 -wd 1e-5 --fp16

#python train.py -m rgb_densenet121 -b 160 -w 24 -s cos --epochs 100 --fine-tune 15 -a safe\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.01 -v -lr 1e-4 --fold 1 --seed 1 -wd 1e-5 --fp16
#
#python train.py -m rgb_densenet121 -b 160 -w 24 -s cos --epochs 100 --fine-tune 15 -a safe\
#  --modification-flag-loss bce 1 --modification -type-loss ce 0.01 -v -lr 1e-4 --fold 2 --seed 2 -wd 1e-5 --fp16

nohup python train.py -m rgb_densenet201 -b 80 -w 24 -s cos --epochs 100 -a safe\
  --modification-flag-loss wbce2 1 --modification-type-loss ce 0.01 -v -lr 3e-4 --fold 3 --seed 3 -wd 1e-5 --fp16 > rgb_densenet201_fold3.log &