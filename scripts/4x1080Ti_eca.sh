export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m ela_ecaresnext26tn_32x4d -b 36 -w 8 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16



python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_swsl_resnext101_32x8d -b 36 -w 8 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16


python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 4 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16

python train.py -m rgb_tf_efficientnet_b6_ns -b 32 -w 4 -d 0.2 -s flat_cos -o Ranger --epochs 50 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 0 -v --fp16 --seed 10 -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/May26_12_01_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/last.pth


python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 1000 -v --fp16 -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/May27_15_25_rgb_tf_efficientnet_b6_ns_fold0_fp16/main/checkpoints/last.pth
