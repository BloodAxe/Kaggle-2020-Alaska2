export KAGGLE_2020_ALASKA2=/data/alaska2

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b7_ns -b 10 -w 6 -d 0.2 -s cos -o fused_sgd --epochs 50 -a medium\
#  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 0 --seed 10 -v --fp16 --obliterate 0.05\
#  -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jun18_14_34_rgb_tf_efficientnet_b7_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best.pth
#
#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b7_ns -b 10 -w 6 -d 0.2 -s cos -o fused_sgd --epochs 50 -a medium\
#  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 1 --seed 20 -v --fp16 --obliterate 0.05
#
#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b7_ns -b 10 -w 6 -d 0.2 -s cos -o fused_sgd --epochs 50 -a medium\
#  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 2 --seed 30 -v --fp16 --obliterate 0.05
#
#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b7_ns -b 10 -w 6 -d 0.2 -s cos -o fused_sgd --epochs 50 -a medium\
#  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 3 --seed 40 -v --fp16 --obliterate 0.05


#python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
#  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 10 -w 6 -s cos -o fused_sgd --epochs 50 -a light\
#  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-3 -wd 1e-4 --fold 1 --seed 20 -v --fp16\
#  -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16/main/checkpoints/last.pth

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
#  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 10 -w 6 -s cos -o fused_sgd --epochs 50\
#  -a hard --modification-flag-loss wbce 1 --modification-type-loss ce 1\
#  -lr 1e-4 -wd 1e-4 --fold 2 --seed 21110 -v --fp16\
#  -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jul10_14_45_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/last.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 10 -w 6 -s cos -o fused_adam --epochs 50\
  -a medium --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-4 -wd 1e-2 --fold 1 --seed 11110 -v --fp16\
  -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jun18_16_07_rgb_tf_efficientnet_b7_ns_fold1_local_rank_0_fp16/main/checkpoints/best.pth