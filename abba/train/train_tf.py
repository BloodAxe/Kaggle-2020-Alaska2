import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import horovod.tensorflow as hvd
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.INFO)
import sys
import numpy as np
import os
from functools import partial
import pickle

sys.path.insert(1,'./')
from train.zoo.models import SR_net_model_eff
from train.datafeeding.input_fn import *
from train.tools.tf_utils import *
from train.tools.kaggle_tools import *
from train.tools.cnn_model_fn import *
from optim import AdamaxOptimizer, get_lr_schedule

def main(unused_argv):
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Train SRNet with tensorflow and horovod")
    arg = parser.add_argument
    arg('--experiment', type=str, default='SRNet_test', help='specific model experiment name')
    arg('--batch-size', type=int, default=32, help='batch size')
    arg('--valid-batch-size', type=int, default=20, help='validation batch size')
    arg('--lr-schedule-id', type=int, default=0, help='learning rate schedule id')
    arg('--warm-start-checkpoint', type=str, default='', help='path to checkpoint')
    arg('--output', type=str, default='weights/', help='output folder')
    arg('--quality-factor', type=int, default=75 , help='quality factor')
    arg('--pair-constraint', type=int, default=1, help='keep pair constraint?')
    
    hvd.init()
    
    args = parser.parse_args()
    classes = ['Cover/', 'JMiPOD/', 'JUNIWARD/', 'UERD/']
    with open('./IL_train_'+str(args.quality_factor)+'.p', 'rb') as handle:
        IL_train = pickle.load(handle)
    with open('./IL_val_'+str(args.quality_factor)+'.p', 'rb') as handle:
        IL_val = pickle.load(handle)

    
    LOG_DIR = args.output+args.experiment+'/'+'QF'+str(args.quality_factor)+'/'
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if hvd.rank() == 0:
        log_args(args.__dict__, os.path.join(LOG_DIR, 'run_hyper_params.txt'))
    
    warm_start_checkpoint = args.warm_start_checkpoint if args.warm_start_checkpoint != '' else None 
    load_checkpoint = None
    lr_schedule = get_lr_schedule(args.lr_schedule_id)
    
    train_interval = 100
    save_interval = 1000
    train_batch_size = args.batch_size
    valid_batch_size = args.valid_batch_size
    num_of_threads = 10
    
    params = dict()
    
    params['scheduler'] = lambda global_step: tf.train.piecewise_constant(global_step, lr_schedule['boundaries'], lr_schedule['values']) 
    
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.allow_growth = True
    
    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = LOG_DIR if hvd.rank() == 0 else None

    # Create the Estimator
    resnet_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir,
        params=params,
        config=tf.estimator.RunConfig(save_summary_steps=save_interval,
                                      save_checkpoints_steps=save_interval,
                                      session_config=config,
                                     keep_checkpoint_max=10000),warm_start_from=warm_start_checkpoint)
    
    # If warm_start, training starts with global_step=0, 
    # use this for loading checkpoints from different folders or for transfer learning.
    # If specific checkpoint is to be loaded, checkpoint file will be updated
    if warm_start_checkpoint is not None:
        start = 0
    elif load_checkpoint == 'last' or load_checkpoint is None:
        start = getLatestGlobalStep(LOG_DIR)
    else:
        start = int(load_checkpoint.split('-')[-1])
        if hvd.rank() == 0:
            updateCheckpointFile(LOG_DIR, load_checkpoint)
    if hvd.rank() == 0:        
        print('global step: ', start)
    
    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
    
    input_fn_train = partial(input_fn, train_batch_size, IL_train, DATA_ROOT_PATH, gen_train, gen_valid, args.pair_constraint, [0.25]*4, classes, num_of_threads, True)
    input_fn_val = partial(input_fn, valid_batch_size, IL_val, DATA_ROOT_PATH, gen_train, gen_valid, args.pair_constraint, [0.25]*4, classes, num_of_threads, False)
            
    for i in range(start, lr_schedule['max_iter'], save_interval):
        resnet_classifier.train(input_fn=input_fn_train,steps=save_interval,hooks=[bcast_hook])
        eval_results = resnet_classifier.evaluate(input_fn=input_fn_val,steps=None,hooks=[bcast_hook])
        print(eval_results)
        
        
    # Delete extra checkpoints
    keepTopKCheckpoints(LOG_DIR)
        
    
if __name__ == '__main__':
    tf.app.run(main)