export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_hrnet18 -b 28 -w 6 -d 0.1 -s flat_cos -o RAdam --epochs 50 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 3e-4 --fold 1 --seed 10000 -v --fp16
