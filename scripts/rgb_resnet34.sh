export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
export CUDA_VISIBLE_DEVICES=0,1,2,3

#python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
#  --modification-flag-loss bce 1 --modification-type-loss ce 0.01\
#  -v -lr 1e-4 --fold 0 --seed 12340

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.01\
  -v -lr 1e-4 --fold 1 --seed 12341

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.01\
  -v -lr 1e-4 --fold 2 --seed 12342

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.01\
  -v -lr 1e-4 --fold 3 --seed 12343
