export KAGGLE_2020_ALASKA2=/data/alaska2

python train.py -m rgb_b0 --size 384 -b 240 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss wbwce 1 --modification-type-loss wce 0.1 -v -lr 1e-4 --fold 0 --seed 12350

python train.py -m rgb_b0 --size 384 -b 240 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss wbwce 1 --modification-type-loss wce 0.1 -v -lr 1e-4 --fold 1 --seed 12351

python train.py -m rgb_b0 --size 384 -b 240 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss wbwce 1 --modification-type-loss wce 0.1 -v -lr 1e-4 --fold 2 --seed 12352

python train.py -m rgb_b0 --size 384 -b 240 -w 24 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss wbwce 1 --modification-type-loss wce 0.1 -v -lr 1e-4 --fold 3 --seed 12353

#
#export KAGGLE_2020_ALASKA2=/data/alaska2
#
#python train.py -m frank -b 128 --size 384 -w 16 -d 0.2 -s simple --epochs 100 --fine-tune 10 -a light\
#  --modification-flag-loss wbwce 1 --modification-type-loss wce 0.01 -v -lr 1e-4 --fold 0 --seed 12350
