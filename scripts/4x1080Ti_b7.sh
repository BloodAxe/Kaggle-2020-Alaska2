export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b7_ns -b 7 -w 6 -d 0.2 -s cos -o fused_sgd --epochs 50 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 3 --seed 40 -v --fp16 --obliterate 0.05
