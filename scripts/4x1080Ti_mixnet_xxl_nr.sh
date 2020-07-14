export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2
#export KAGGLE_2020_ISTEGO100K=/home/bloodaxe/datasets/istego100k

#python -m torch.distributed.launch --nproc_per_node=4 train_d_paired.py\
#  -m nr_rgb_mixnet_xxl -b 10 -w 6 -d 0.2 -s cos -o SGD --epochs 75 -a light\
#  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 3e-3 -wd 1e-5 -v --fp16\
#  --fold 1 --seed 101

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
#  -m nr_rgb_mixnet_xxl -b 10 -w 6 -d 0.2 -s cos -o SGD --epochs 75 -a medium\
#  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 3e-3 -wd 1e-5 -v --fp16\
#  --fold 1 --seed 101\
#  -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jul08_13_05_nr_rgb_mixnet_xxl_fold1_paired_local_rank_0_fp16/main/checkpoints_auc_classifier/train.1.pth

#python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
#  -m nr_rgb_mixnet_xxl -b 10 -w 6 -s cos -o SGD --epochs 75 -a medium\
#  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 8e-5 -wd 1e-5 -v --fp16\
#  --fold 1 --seed 40101\
#  -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jul09_23_37_nr_rgb_mixnet_xxl_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_mixnet_xxl -b 10 -w 6 -s flat_cos -o adamw --epochs 25 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 -v --fp16\
  --fold 1 --seed 400101\
  -c /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/Jul13_22_34_nr_rgb_mixnet_xxl_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth