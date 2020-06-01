export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m dct_seresnext50 -b 64 -w 8 -d 0.25 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 1000 -v --fp16\
  -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/May31_15_04_dct_seresnext50_fold0_local_rank_0_fp16/main/checkpoints/best.pth