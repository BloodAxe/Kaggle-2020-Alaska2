export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m ela_tf_efficientnet_b6_ns\
  -b 8 -w 6 -d 0.25 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 5e-3 -wd 1e-4 --fold 0 --seed 10 -v --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m ela_tf_efficientnet_b6_ns\
  -b 8 -w 6 -d 0.25 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 5e-3 -wd 1e-4 --fold 1 --seed 11 -v --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/models/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m ela_tf_efficientnet_b6_ns\
  -b 8 -w 6 -d 0.25 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 5e-3 -wd 1e-4 --fold 2 --seed 12 -v --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/models/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m ela_tf_efficientnet_b6_ns\
  -b 8 -w 6 -d 0.25 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 5e-3 -wd 1e-4 --fold 3 --seed 13 -v --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/models/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints/best.pth
