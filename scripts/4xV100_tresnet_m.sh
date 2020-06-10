export KAGGLE_2020_ALASKA2=/data/alaska2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.5 -s flat_cos2 -o Ranger --epochs 150 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 0 --seed 0 -v --fp16 -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jun09_21_05_rgb_tresnet_m_448_fold0_local_rank_0_fp16/main/checkpoints_auc/last.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.5 -s cos -o fused_sgd --epochs 30 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 -wd 1e-4 --fold 1 --seed 1111 -v --fp16 -c /home/ubuntu/code/Kaggle-2020-Alaska2/runs/Jun09_11_41_rgb_tresnet_m_448_fold1_local_rank_0_fp16/main/checkpoints_auc_classifier/best.pth

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.5 -s flat_cos2 -o RAdam --epochs 150 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 2 --seed 2 -v --fp16

python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_tresnet_m_448 -b 92 -w 8 -d 0.5 -s flat_cos2 -o fused_lamb --epochs 150 -a light\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 -wd 1e-4 --fold 3 --seed 3 -v --fp16
