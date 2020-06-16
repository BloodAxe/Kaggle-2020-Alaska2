export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

# Only residual input
#python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m res_tf_efficientnet_b2_ns -b 24 -w 8 -d 0.2 -s cos -o SGD --epochs 50 -a medium\
#  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 --fold 0 --seed 10000 -v --fp16

# RGB + residual input
python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_res_tf_efficientnet_b2_ns -b 24 -w 8 -d 0.2 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 --fold 0 --seed 10000 -v --fp16

# RGB + residual with late concat
python -m torch.distributed.launch --nproc_per_node=4 train_d.py -m rgb_res_sms_tf_efficientnet_b2_ns -b 12 -w 8 -d 0.2 -s cos -o SGD --epochs 50 -a medium\
  --modification-flag-loss bce 1 --modification-type-loss ce 1 -lr 1e-2 --fold 0 --seed 10000 -v --fp16
