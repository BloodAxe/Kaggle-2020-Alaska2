export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 --embedding-loss ccos 1.0\
  -v -lr 1e-4 --fold 0 --seed 12340 --balance\
  -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/May07_12_52_rgb_resnet34_fold0/main/checkpoints/last.pth