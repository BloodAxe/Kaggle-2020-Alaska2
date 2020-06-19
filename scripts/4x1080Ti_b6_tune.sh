export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
export KAGGLE_2020_ISTEGO100K=/home/bloodaxe/datasets/istego100k

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s flat_cos -o SGD --epochs 50 -a medium\
#  --modification-flag-loss wbce 1 --modification-type-loss focal 1 -lr 1e-3 --fold 0 --seed 100 -v --fp16\
#  -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/models/Jun05_08_49_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d_extra_data.py -m rgb_tf_efficientnet_b6_ns -b 8 -w 6 -d 0.2 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss focal 1 -lr 1e-3 --fold 0 --seed 100 -v --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jun18_19_24_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16/main/checkpoints_auc/best.pth

