export KAGGLE_2020_ALASKA2=/data/alaska2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_qf_swsl_resnext101_32x8d -b 20 -w 6 -d 0.2 -s cos -o fused_sgd --epochs 150 -a medium\
  --modification-flag-loss binary_focal 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 3 --seed 100030 -v --fp16 --obliterate 0.15