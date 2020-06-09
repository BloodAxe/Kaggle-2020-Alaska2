export KAGGLE_2020_ALASKA2=/data/alaska2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 72 -w 8 -d 0.2 -s flat_cos -o fused_sgd --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 --fold 1 --seed 1 -v --fp16
