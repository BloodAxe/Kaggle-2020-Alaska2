export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
#export KAGGLE_2020_ISTEGO100K=/home/bloodaxe/datasets/istego100k

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns_mish_gep -b 8 -w 6 -s cos -o adamw --epochs 25 -a medium\
  --modification-flag-loss wbce 0.1 --embedding-loss arc_face 1 -lr 3e-4 -wd 1e-2 -v --fp16\
  --fold 0 --seed 100\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jul15_00_15_nr_rgb_tf_efficientnet_b6_ns_mish_gep_fold0_local_rank_0_fp16/main/checkpoints_auc_embedding/last.pth