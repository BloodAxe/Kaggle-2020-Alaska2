export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m ela_ecaresnext26tn_32x4d -b 36 -w 8 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16



python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_swsl_resnext101_32x8d -b 36 -w 8 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 4 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16
