export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d_paired.py -m bit_m_rx50_1 -b 16 -w 6 -d 0.2 -s cos -o SGD --epochs 75 -a light\
  --modification-flag-loss bce 1 -lr 1e-2 --fold 0 -v --bitmix

python -m torch.distributed.launch --nproc_per_node=4 train_d_paired.py -m rgb_tf_efficientnet_b2_ns -b 12 -w 6 -d 0.2 -s cos -o SGD --epochs 75 -a light\
  --modification-flag-loss bce 1 -lr 1e-2 --fold 0 -v --bitmix
