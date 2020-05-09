export KAGGLE_2020_ALASKA2=/data/alaska2

python train2.py -m rgb_seresnext50 -b 16 -w 24 -d 0.5 -s cos --epochs 20 -a safe\
  --modification-flag-loss wbce 1 --modification-type-loss ce 0.01 -v -lr 1e-4 --fold 0 --seed 22350

python train2.py -m rgb_seresnext50 -b 16 -w 24 -d 0.5 -s cos --epochs 20 -a safe\
  --modification-flag-loss wbce 1 --modification-type-loss ce 0.01 --embedding-loss cntr 1.0 -v -lr 1e-4 --fold 0 --seed 22350

python train2.py -m rgb_seresnext50 -b 16 -w 24 -d 0.5 -s cos --epochs 20 -a safe\
  --modification-flag-loss wbce 0.01 --embedding-loss cntrv2 1.0 -v -lr 1e-4 --fold 0 --seed 22350


python train2.py -m rgb_seresnext50 --size 384 -b 32 -w 24 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss wbce 1 --modification-type-loss ce 0.01 -v -lr 1e-4 --fold 1 --seed 22351

python train2.py -m rgb_seresnext50 --size 384 -b 32 -w 24 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss wbce 1 --modification-type-loss ce 0.01 -v -lr 1e-4 --fold 2 --seed 22352

python train2.py -m rgb_seresnext50 --size 384 -b 32 -w 24 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss wbce 1 --modification-type-loss ce 0.01 -v -lr 1e-4 --fold 3 --seed 22353