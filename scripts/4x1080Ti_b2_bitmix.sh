export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b2_ns -b 8 -w 6 -d 0.2 -s cos -o SGD --epochs 75 -a medium\
  --modification-flag-loss bce 1 -lr 1e-2 --fold 0 --seed 10000 -v --bitmix
