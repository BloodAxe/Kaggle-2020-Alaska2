export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m dct_efficientnet_b6\
  -b 64 -w 8 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16