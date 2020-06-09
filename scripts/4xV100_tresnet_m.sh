export KAGGLE_2020_ALASKA2=/data/alaska2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.2 -s flat_cos -o Ranger --epochs 50 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 0 --seed 0 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.2 -s flat_cos -o fused_sgd --epochs 50 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 1 --seed 1 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.2 -s flat_cos -o RAdam --epochs 50 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 2 --seed 2 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.2 -s flat_cos -o fused_lamb --epochs 50 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 3 --seed 3 -v --fp16
