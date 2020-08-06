export DATA_ROOT_PATH=/media/alaska2/all_qfs/
cd abba/

python3 download_splits.py
# Training ImageNet pretrained models
# All these scripts fit in a Titan RTX GPU, please adjust the batch size/learning rate (at your own risk)
# If you have multiple GPUs you might want to run these commands by changing the device id

# python3 train/train_pytorch.py --model mixnet_s --experiment mixnet_S_R --fp16 0 --decoder R --batch-size 16
python3 train/train_pytorch.py --model mixnet_s --experiment mixnet_S_NR_mish --batch-size 32
python3 train/train_pytorch.py --model mixnet_xl --experiment mixnet_xL_NR_mish --batch-size 32
# python3 train/train_pytorch.py --model efficientnet-b2 --experiment efficientnet_b2_R --fp16 0 --decoder R --batch-size 16
python3 train/train_pytorch.py --model efficientnet-b2 --experiment efficientnet_b2_NR_mish --batch-size 32
python3 train/train_pytorch.py --model efficientnet-b4 --experiment efficientnet_b4_NR_mish --batch-size 32
python3 train/train_pytorch.py --model efficientnet-b5 --experiment efficientnet_b5_NR_mish --batch-size 32
# python3 train/train_pytorch.py --model efficientnet-b6 --experiment efficientnet_b6_NR_mish --batch-size 29

# Training SRNet
# This will use 2 GPUs available using horovod
# You can scale to more GPUs or multiple machines to speed up the training
# adjust the batch size/learning rate if needed (at your own risk)
# mpirun -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca plm_rsh_args "-p 22" python3 train/train_tf.py

horovodrun -np 2 -H localhost:2 python3 train/train_tf.py --quality-factor 75
horovodrun -np 2 -H localhost:2 python3 train/train_tf.py --warm-start-checkpoint weights/SRNet/QF75/model.ckpt-232000 --quality-factor 90 --lr-schedule-id 1
horovodrun -np 2 -H localhost:2 python3 train/train_tf.py --warm-start-checkpoint weights/SRNet/QF75/model.ckpt-232000 --quality-factor 95 --lr-schedule-id 1

# No pair-constraint training, used for WIFS2020 submission
# horovodrun -np 2 -H localhost:2 python3 train/train_tf.py --warm-start-checkpoint --quality-factor 75 --pair-constraint 0 --lr-schedule-id 2
# horovodrun -np 2 -H localhost:2 python3 train/train_tf.py --warm-start-checkpoint --quality-factor 95 --pair-constraint 0 --lr-schedule-id 2
# horovodrun -np 2 -H localhost:2 python3 train/train_tf.py --warm-start-checkpoint --quality-factor 90 --pair-constraint 0 --lr-schedule-id 2

# Generating Rich Models features
# Please make sure your specs can handle 24 parallel jobs
# This might take more than a day, we advise to use scripts from http://dde.binghamton.edu/download/feature_extractors/ And run on a cluster with multiple nodes, most research groups have access to such compute power. 
# We used the feature extractors DCTR and JRM on the luminance channel only 

python3 rich_models/generate_features.py --quality-factor 75 --model DCTR --folder Cover/ & 
python3 rich_models/generate_features.py --quality-factor 90 --model DCTR --folder Cover/ & 
python3 rich_models/generate_features.py --quality-factor 95 --model DCTR --folder Cover/ & 
python3 rich_models/generate_features.py --quality-factor 75 --model DCTR --folder JUNIWARD/ &
python3 rich_models/generate_features.py --quality-factor 90 --model DCTR --folder JUNIWARD/ &
python3 rich_models/generate_features.py --quality-factor 95 --model DCTR --folder JUNIWARD/ &
python3 rich_models/generate_features.py --quality-factor 75 --model DCTR --folder UERD/ &
python3 rich_models/generate_features.py --quality-factor 90 --model DCTR --folder UERD/ &
python3 rich_models/generate_features.py --quality-factor 95 --model DCTR --folder UERD/ &
python3 rich_models/generate_features.py --quality-factor 75 --model DCTR --folder JMiPOD/ &
python3 rich_models/generate_features.py --quality-factor 90 --model DCTR --folder JMiPOD/ &
python3 rich_models/generate_features.py --quality-factor 95 --model DCTR --folder JMiPOD/ &
python3 rich_models/generate_features.py --quality-factor 75 --model JRM --folder Cover/ &
python3 rich_models/generate_features.py --quality-factor 90 --model JRM --folder Cover/ &
python3 rich_models/generate_features.py --quality-factor 95 --model JRM --folder Cover/ &
python3 rich_models/generate_features.py --quality-factor 75 --model JRM --folder JUNIWARD/ &
python3 rich_models/generate_features.py --quality-factor 90 --model JRM --folder JUNIWARD/ &
python3 rich_models/generate_features.py --quality-factor 95 --model JRM --folder JUNIWARD/ &
python3 rich_models/generate_features.py --quality-factor 75 --model JRM --folder UERD/ &
python3 rich_models/generate_features.py --quality-factor 90 --model JRM --folder UERD/ &
python3 rich_models/generate_features.py --quality-factor 95 --model JRM --folder UERD/ &
python3 rich_models/generate_features.py --quality-factor 75 --model JRM --folder JMiPOD/ &
python3 rich_models/generate_features.py --quality-factor 90 --model JRM --folder JMiPOD/ &
python3 rich_models/generate_features.py --quality-factor 95 --model JRM --folder JMiPOD/ 

# Train FLD ensemble for each quality factor and each rich model
# Used scripts from http://dde.binghamton.edu/download/ensemble/
python3 rich_models/train_ensemble.py --quality-factor 95 --model JRM &
python3 rich_models/train_ensemble.py --quality-factor 90 --model JRM &
python3 rich_models/train_ensemble.py --quality-factor 75 --model JRM &
python3 rich_models/train_ensemble.py --quality-factor 95 --model DCTR &
python3 rich_models/train_ensemble.py --quality-factor 90 --model DCTR &
python3 rich_models/train_ensemble.py --quality-factor 75 --model DCTR 