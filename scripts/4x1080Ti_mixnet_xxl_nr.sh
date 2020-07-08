export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
#export KAGGLE_2020_ISTEGO100K=/home/bloodaxe/datasets/istego100k

python -m torch.distributed.launch --nproc_per_node=4 train_d_paired.py\
  -m nr_rgb_mixnet_xxl -b 8 -w 6 -d 0.2 -s cos -o SGD --epochs 75 -a light\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-5 -v --fp16\
  --fold 1 --seed 101
