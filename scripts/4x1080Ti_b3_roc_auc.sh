export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b3_ns -b 20 -w 6 -d 0.2 -s flat_cos -o Ranger --epochs 50 -a hard\
#  --modification-flag-loss roc_auc 1 --modification-type-loss ce 1  -lr 1e-3 --fold 0 --seed 10000 -v --fp16 --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun27_20_18_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/last.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b3_ns -b 20 -w 6 -d 0.2 -s flat_cos -o SGD --epochs 50 -a hard\
  --modification-flag-loss roc_auc 1 --modification-type-loss ce 1  -lr 1e-3 --fold 0 --seed 10000 -v --fp16\
  --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun27_21_24_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b3_ns -b 20 -w 6 -d 0.2 -s flat_cos -o SGD --epochs 50 -a hard\
#  --modification-flag-loss roc_auc 1 --modification-type-loss ce 1  -lr 1e-3 --fold 1 --seed 10001 -v --fp16\
#  --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun29_06_28_rgb_tf_efficientnet_b3_ns_fold1_local_rank_0_fp16/main/checkpoints_auc/last.pth
#
#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b3_ns -b 20 -w 6 -d 0.2 -s flat_cos -o Ranger --epochs 100 -a hard\
#  --modification-flag-loss roc_auc 1 --modification-type-loss ce 1  -lr 1e-3 --fold 2 --seed 10002 -v --fp16
#
#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b3_ns -b 20 -w 6 -d 0.2 -s flat_cos -o Ranger --epochs 100 -a hard\
#  --modification-flag-loss roc_auc 1 --modification-type-loss ce 1 -lr 1e-3 --fold 3 --seed 10003 -v --fp16
