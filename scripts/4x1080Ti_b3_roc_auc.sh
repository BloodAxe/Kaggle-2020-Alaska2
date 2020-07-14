export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m rgb_tf_efficientnet_b3_ns -b 20 -w 6 -s flat_cos -o adamw --epochs 50 -a medium\
  --bitmix --modification-flag-loss bce 1 --modification-type-loss roc_auc_ce 0.1 -lr 1e-4 -wd 1e-2 --fold 0 --seed 10000 -v --fp16\
  --checkpoint /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/F_Jun29_19_43_rgb_tf_efficientnet_b3_ns_fold0_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth