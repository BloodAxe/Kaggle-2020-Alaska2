export KAGGLE_2020_ALASKA2=/data/alaska2

# pip install git+https://github.com/dwgoon/jpegio
# Extract DCT and save to NPZ array to speed up reading and
# prevent memory leak in jpegio to eat all memory during training
python save_dct.py -f Cover -p 0
python save_dct.py -f JMiPOD -p 0
python save_dct.py -f JUNIWARD -p 0
python save_dct.py -f UERD -p 0
python save_dct.py -f Cover -p 1
python save_dct.py -f JMiPOD -p 1
python save_dct.py -f JUNIWARD -p 1
python save_dct.py -f UERD -p 1
python save_dct.py -f Test

python -m torch.distributed.launch --nproc_per_node=3 train_d.py\
  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 6 -w 6 -s cos -o fused_adam --epochs 75\
  -a medium --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-4 -wd 1e-2 --fold 0 --seed 11110 -v --fp16

python -m torch.distributed.launch --nproc_per_node=3 train_d.py\
  -m nr_rgb_tf_efficientnet_b7_ns_mish -b 6 -w 6 -s cos -o fused_adam --epochs 75\
  -a medium --modification-flag-loss wbce 1 --modification-type-loss ce 1\
  -lr 1e-4 -wd 1e-2 --fold 3 --seed 31110 -v --fp16