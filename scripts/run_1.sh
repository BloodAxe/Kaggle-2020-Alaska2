export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-4 --fold 0 --seed 12340 --balance -c runs/May06_21_59_rgb_resnet34_fold0_fp16/main/checkpoints/best.pth

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-4 --fold 1 --seed 12341 --balance -c runs/May06_21_59_rgb_resnet34_fold1_fp16/main/checkpoints/best.pth

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-4 --fold 2 --seed 12342 --balance -c runs/May06_21_59_rgb_resnet34_fold2_fp16/main/checkpoints/best.pth

python train.py -m rgb_resnet34 -b 184 -w 16 -d 0.2 -s cos --epochs 100 --fine-tune 10 -a safe\
  --modification-flag-loss bce 1 --modification-type-loss ce 0.1 -v -lr 1e-4 --fold 3 --seed 12343 --balance -c runs/May06_21_59_rgb_resnet34_fold3_fp16/main/checkpoints/best.pth
