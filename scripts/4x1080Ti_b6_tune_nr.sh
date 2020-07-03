export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
#export KAGGLE_2020_ISTEGO100K=/home/bloodaxe/datasets/istego100k

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s cos -o SGD --epochs 15 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-3 -v --fp16\
  --fold 0 --seed 100\
  --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s cos -o SGD --epochs 15 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-3 -v --fp16\
  --fold 1 --seed 101\
  --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun09_16_38_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s cos -o SGD --epochs 15 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-3 -v --fp16\
  --fold 2 --seed 102\
  --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun11_08_51_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s cos -o SGD --epochs 15 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-3 -v --fp16\
  --fold 3 --seed 103\
  --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun11_18_38_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth
