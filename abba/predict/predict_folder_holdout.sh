export DATA_ROOT_PATH=/media/alaska2/all_qfs/
export folders='Cover/ JUNIWARD/ JMiPOD/ UERD/'
export splits='val test'

for split in $splits
do
    python3 predict/predict_folder_richmodels.py --folder 'Cover/' --subset $split --experiment JRM --quality-factor 75 --checkpoint weights/rich_models/QF75_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'Cover/' --subset $split --experiment JRM --quality-factor 90 --checkpoint weights/rich_models/QF90_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'Cover/' --subset $split --experiment JRM --quality-factor 95 --checkpoint weights/rich_models/QF95_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'Cover/' --subset $split --experiment DCTR --quality-factopr 75 --checkpoint weights/rich_models/QF75_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'Cover/' --subset $split --experiment DCTR --quality-factor 90 --checkpoint weights/rich_models/QF90_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'Cover/' --subset $split --experiment DCTR --quality-factor 95 --checkpoint weights/rich_models/QF95_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JUNIWARD/' --subset $split --experiment JRM --quality-factor 75 --checkpoint weights/rich_models/QF75_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JUNIWARD/' --subset $split --experiment JRM --quality-factor 90 --checkpoint weights/rich_models/QF90_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JUNIWARD/' --subset $split --experiment JRM --quality-factor 95 --checkpoint weights/rich_models/QF95_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JUNIWARD/' --subset $split --experiment DCTR --quality-factor 75 --checkpoint weights/rich_models/QF75_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JUNIWARD/' --subset $split --experiment DCTR --quality-factor 90 --checkpoint weights/rich_models/QF90_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JUNIWARD/' --subset $split --experiment DCTR --quality-factor 95 --checkpoint weights/rich_models/QF95_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JMiPOD/' --subset $split --experiment JRM --quality-factor 75 --checkpoint weights/rich_models/QF75_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JMiPOD/' --subset $split --experiment JRM --quality-factor 90 --checkpoint weights/rich_models/QF90_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JMiPOD/' --subset $split --experiment JRM --quality-factor 95 --checkpoint weights/rich_models/QF95_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JMiPOD/' --subset $split --experiment DCTR --quality-factor 75 --checkpoint weights/rich_models/QF75_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JMiPOD/' --subset $split --experiment DCTR --quality-factor 90 --checkpoint weights/rich_models/QF90_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'JMiPOD/' --subset $split --experiment DCTR --quality-factor 95 --checkpoint weights/rich_models/QF95_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'UERD/' --subset $split --experiment JRM --quality-factor 75 --checkpoint weights/rich_models/QF75_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'UERD/' --subset $split --experiment JRM --quality-factor 90 --checkpoint weights/rich_models/QF90_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'UERD/' --subset $split --experiment JRM --quality-factor 95 --checkpoint weights/rich_models/QF95_JRM_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'UERD/' --subset $split --experiment DCTR --quality-factor 75 --checkpoint weights/rich_models/QF75_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'UERD/' --subset $split --experiment DCTR --quality-factor 90 --checkpoint weights/rich_models/QF90_DCTR_Y_ensemble_v7.mat &
    python3 predict/predict_folder_richmodels.py --folder 'UERD/' --subset $split --experiment DCTR --quality-factor 95 --checkpoint weights/rich_models/QF95_DCTR_Y_ensemble_v7.mat 
        
    for folder in $folders
    do
        python3 predict/predict_folder_outofbounds.py --folder $folder --subset $split 
        python3 predict/predict_folder_tf.py --folder $folder --subset $split --quality-factor 75 --checkpoint weights/SRNet/QF75/model.ckpt-232000
        python3 predict/predict_folder_tf.py --folder $folder --subset $split --quality-factor 90 --checkpoint weights/SRNet/QF90/model.ckpt-39000
        python3 predict/predict_folder_tf.py --folder $folder --subset $split --quality-factor 95 --checkpoint weights/SRNet/QF95/model.ckpt-18000
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model efficientnet-b4 --experiment efficientnet_b4_NR_mish --checkpoint weights/efficientnet_b4_NR_mish/best-checkpoint-017epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model efficientnet-b5 --experiment efficientnet_b5_NR_mish --checkpoint weights/efficientnet_b5_NR_mish/best-checkpoint-018epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model mixnet_xl --experiment mixnet_xL_NR_mish --surgery 1 --checkpoint weights/mixnet_xL_NR_mish/best-checkpoint-021epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model efficientnet-b2 --experiment efficientnet_b2_NR --surgery 0 --fp16 0 --checkpoint weights/efficientnet_b2/NR/best-checkpoint-028epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model efficientnet-b2 --experiment efficientnet_b2_R --decoder R --surgery 0 --fp16 0 --checkpoint weights/efficientnet_b2/R/best-checkpoint-028epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed0 --decoder R --surgery 0 --fp16 0 --checkpoint weights/mixnet_S/R_seed0/best-checkpoint-033epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed1 --decoder R --surgery 0 --fp16 0 --checkpoint weights/mixnet_S/R_seed1/best-checkpoint-035epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed2 --decoder R --surgery 0 --fp16 0 --checkpoint weights/mixnet_S/R_seed2/best-checkpoint-036epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed3 --decoder R --surgery 0 --fp16 0 --checkpoint weights/mixnet_S/R_seed3/best-checkpoint-038epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed4 --decoder R --surgery 0 --fp16 0 --checkpoint weights/mixnet_S/R_seed4/best-checkpoint-035epoch.bin
        python3 predict/predict_folder_pytorch.py --folder $folder --subset $split --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_NR --surgery 0 --fp16 0 --checkpoint weights/mixnet_S/NR/best-checkpoint-058epoch.bin
        python3 predict/group_experiments.py --folder $folder --subset $split
    done
done