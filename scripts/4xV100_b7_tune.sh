export KAGGLE_2020_ALASKA2=/data/alaska2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 8 -w 6 -s cos -o fused_adam --epochs 10\
  -a hard --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-4 -wd 1e-2 --fold 1 --seed 11110 -v --fp16\
  -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jul12_18_42_nr_rgb_tf_efficientnet_b7_ns_mish_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 8 -w 6 -s cos -o fused_adam --epochs 10\
  -a hard --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-4 -wd 1e-2 --fold 2 --seed 11110 -v --fp16\
  -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jul11_16_37_nr_rgb_tf_efficientnet_b7_ns_mish_fold2_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth
