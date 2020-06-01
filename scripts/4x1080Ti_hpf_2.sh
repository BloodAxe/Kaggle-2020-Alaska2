export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m dct_seresnext50 -b 64 -w 8 -d 0.5 -s flat_cos -o Ranger --epochs 75 -a none\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 1000 -v --fp16