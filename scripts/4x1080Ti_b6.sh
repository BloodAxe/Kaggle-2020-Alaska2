export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s flat_cos -o Ranger --epochs 75 -a medium\
#  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 10000 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.3 -s cos -o SGD --epochs 25 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 1 --seed 10001 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.3 -s cos -o SGD --epochs 25 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 2 --seed 10002 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.3 -s cos -o SGD --epochs 25 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 3 --seed 10003 -v --fp16


#python oof_predictions.py /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best.pth /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best.pth /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth -b 32 -w 4; python predict.py /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best.pth /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best.pth /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth -b 32 -w 4