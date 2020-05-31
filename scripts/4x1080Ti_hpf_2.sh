export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

python train.py -m hpf_net2 -b 96 -w 24 -d 0.2 -s flat_cos -o Ranger --epochs 25 -a light\
  --modification-flag-loss rank2 1\
  --modification-type-loss ce 0.1\
  -lr 1e-4 --fold 1 --seed 111 --fp16\
  --transfer /home/bloodaxe/develop/Kaggle-2020-Alaska2/runs/May21_10_14_hpf_net_fold0_fp16/main/checkpoints_auc_classifier/best.pth


python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m dct_seresnext50 -b 256 -w 4 -d 0.5 -s flat_cos -o Ranger --epochs 75 -a none\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-3 --fold 0 --seed 1000 -v --fp16