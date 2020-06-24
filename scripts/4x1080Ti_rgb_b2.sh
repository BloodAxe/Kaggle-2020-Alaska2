export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b2_ns_avgmax -b 24 -w 8 -d 0.25 -s cos -o SGD --epochs 50 -a hard\
  --modification-flag-loss wbce 1 --modification-type-loss focal 1 -lr 1e-2 -wd 1e-5 --fold 0 --seed 10000 -v --fp16