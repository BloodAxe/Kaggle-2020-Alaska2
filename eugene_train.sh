# Environment variable KAGGLE_2020_ALASKA2 must be set prior running this script:
# export KAGGLE_2020_ALASKA2=/home/bloodaxe/datasets/ALASKA2

# This script assumes it is running on 4 GPU setup like 4xV100 (with 16Gb VRAM) or 1080Ti/2080Ti (11Gb VRAM).
# In case of 11Gb VRAM per card you should lower batch size to 6 for B7 models by adjusting value of `-b` flag to `-b 6`

# Train 4 folds of B6-NR
python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 8 -s cos -o adamw --epochs 100 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 -v --fp16\
  --fold 0 -x G_nr_rgb_tf_efficientnet_b6_ns_fold0

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 8 -s cos -o adamw --epochs 100 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 -v --fp16\
  --fold 1 -x G_nr_rgb_tf_efficientnet_b6_ns_fold1

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 8 -s cos -o adamw --epochs 100 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 -v --fp16\
  --fold 2 -x G_nr_rgb_tf_efficientnet_b6_ns_fold2

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns -b 8 -w 8 -s cos -o adamw --epochs 100 -a medium\
  --modification-flag-loss wbce 1 --modification-type-loss ce 1 -lr 1e-4 -wd 1e-2 -v --fp16\
  --fold 3 -x G_nr_rgb_tf_efficientnet_b6_ns_fold3

# Train B7-NR+Mish
python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 8 -w 6 -s cos -o fused_adam --epochs 10\
  -a medium --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-4 -wd 1e-2 --fold 1 --seed 11110 -v --fp16\
  --fold 1 -x H_nr_rgb_tf_efficientnet_b7_ns_mish_fold1

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 8 -w 6 -s cos -o fused_adam --epochs 10\
  -a medium --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-4 -wd 1e-2 --seed 11110 -v --fp16\
   --fold 2 -x H_nr_rgb_tf_efficientnet_b7_ns_mish_fold2

# Fine-tune two folds B6-NR+Mish
python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns_mish -b 8 -w 6 -s cos -o fused_adam --epochs 10\
  -a medium --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-3 -wd 1e-2 --fold 1 --seed 11110 -v --fp16\
  --fold 0 -x K_nr_rgb_tf_efficientnet_b6_ns_mish_fold0

python -m torch.distributed.launch --nproc_per_node=4 train_d.py\
  -m nr_rgb_tf_efficientnet_b6_ns_mish -b 8 -w 6 -s cos -o fused_adam --epochs 10\
  -a medium --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-3 -wd 1e-2 --fold 1 --seed 11110 -v --fp16\
  --fold 3 -x K_nr_rgb_tf_efficientnet_b6_ns_mish_fold3