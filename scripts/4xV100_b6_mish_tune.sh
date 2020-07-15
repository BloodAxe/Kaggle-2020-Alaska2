export KAGGLE_2020_ALASKA2=/data/alaska2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns_mish_gep -b 12 -w 6 -s cos -o fused_adam --epochs 25 -a medium\
  --modification-flag-loss wbce --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 --fold 0 --seed 0 -v --fp16\
  --transfer /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jul03_21_14_nr_rgb_tf_efficientnet_b6_ns_fold0_local_rank_0_fp16.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns_mish_gep -b 12 -w 6 -s cos -o fused_adam --epochs 25 -a medium\
  --modification-flag-loss wbce --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 --fold 1 --seed 1 -v --fp16\
  --transfer /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jul05_00_24_nr_rgb_tf_efficientnet_b6_ns_fold1_local_rank_0_fp16.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns_mish_gep -b 12 -w 6 -s cos -o fused_adam --epochs 25 -a medium\
  --modification-flag-loss wbce --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 --fold 2 --seed 2 -v --fp16\
  --transfer /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jul06_03_39_nr_rgb_tf_efficientnet_b6_ns_fold2_local_rank_0_fp16.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns_mish_gep -b 12 -w 6 -s cos -o fused_adam --epochs 25 -a medium\
  --modification-flag-loss wbce --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 --fold 3 --seed 3 -v --fp16\
  --transfer /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jul07_06_38_nr_rgb_tf_efficientnet_b6_ns_fold3_local_rank_0_fp16.pth
  
