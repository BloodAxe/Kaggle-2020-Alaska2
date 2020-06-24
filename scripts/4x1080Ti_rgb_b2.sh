export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d_paired.py\
  -m rgb_tf_efficientnet_b2 -b 28 -w 8 -d 0.25 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss rank2 1 --modification-type-loss roc_auc_ce 1\
  -lr 1e-2 -wd 1e-5 --fold 2 --seed 10000 -v --fp16\
  -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun02_12_26_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16/Jun02_12_26_rgb_tf_efficientnet_b2_ns_fold2_local_rank_0_fp16.pth

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b2_ns_avgmax -b 28 -w 8 -d 0.25 -s cos -o SGD --epochs 50 -a hard\
#  --modification-flag-loss wbce 1 --modification-type-loss focal 1 -lr 1e-2 -wd 1e-5 --fold 0 --seed 10000 -v --fp16