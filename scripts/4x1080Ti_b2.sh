export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b2_ns -b 16 -w 6 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a hard\
  --modification-flag-loss soft_bce 1 --modification-type-loss soft_ce 1 -lr 1e-3 --fold 2 --seed 20000 -v --fp16
