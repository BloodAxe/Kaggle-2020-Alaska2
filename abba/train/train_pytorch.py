import argparse
from torch.utils.data.sampler import SequentialSampler
import sys
import numpy as np
import os
import sys
import pandas as pd
import pickle
from apex import amp
sys.path.insert(1,'./')
from train.zoo.models import *
from train.zoo.surgery import *
from train.datafeeding.retriever import *
from train.tools.torch_utils import *
from train.tools.fitter import *
from train.optim import get_optimizer

def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Train imagenet pretrained model using pytorch")
    arg = parser.add_argument
    arg('--model', type=str, default='mixnet_s', help='model name')
    arg('--experiment', type=str, default='test', help='specific model experiment name')
    arg('--surgery', type=int, default=1, help='modification level')
    arg('--optimizer-name', type=str, default='adamw', help='optimizer name')
    arg('--start-lr', type=float, default=1e-3, help='starting learning rate')
    arg('--weight-decay', type=float, default=1e-2, help='weight decay')
    arg('--num-epochs', type=int, default=40, help='number of training epochs')
    arg('--batch-size', type=int, default=16, help='batch size')
    arg('--load-checkpoint', type=str, default='' , help='path to checkpoint to load')
    arg('--fine-tune', type=int, default=0, help='Finetune? ie reset optimization parameters and load only weights')
    arg('--output', type=str, default='weights/', help='output folder')
    arg('--random-seed', type=int, default=0, help='random seed')
    arg('--fp16', type=int, default=1, help='use AMP?')
    arg('--decoder', type=str, default='NR', help='jpeg decoder, R or NR')
    arg('--device', type=str, default='0', help='device id')
    
    
    args = parser.parse_args()
    
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    #torch.cuda.set_device(int(args.device.split(':')[-1]))
    
    seed_everything(args.random_seed)
    device = 'cuda:0' #torch.device(args.device)
    QFs = ['75','90', '95']
    Classes = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
    IL_train = []
    IL_val = []
    
    for QF in QFs:
        with open('./IL_train_'+QF+'.p', 'rb') as handle:
            IL_train.extend(pickle.load(handle))
        with open('./IL_val_'+QF+'.p', 'rb') as handle:
            IL_val.extend(pickle.load(handle))
            
    dataset = []
    for label, kind in enumerate(Classes):
        for path in IL_train:
            dataset.append({
                'kind': kind,
                'image_name': path,
                'label': label,
                'fold':1,
            })
    for label, kind in enumerate(Classes):
        for path in IL_val:
            dataset.append({
                'kind': kind,
                'image_name': path,
                'label': label,
                'fold':0,
            })
            
    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)
   
    class TrainGlobalConfig:
        base_dir = args.output+args.experiment
        num_workers = 5
        batch_size = args.batch_size 
        n_epochs = args.num_epochs
        optimizer = get_optimizer(args.optimizer_name)
        lr = args.start_lr
        weight_decay = args.weight_decay
        keep_top = 3
        if args.fp16:
            loss_scale = 'dynamic'
            opt_level = 'O1'
        else:
            loss_scale = 1.0
            opt_level = 'O0'
        fine_tune = args.fine_tune
        verbose = True
        verbose_step = 1    
        SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_params = dict(
            mode='min',
            factor=0.5,
            patience=1,
            verbose=False, 
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0, 
            min_lr=1e-8,
            eps=1e-08
        )

    os.makedirs(TrainGlobalConfig.base_dir, exist_ok=True)
    log_args(args.__dict__, os.path.join(args.output+args.experiment, 'run_hyper_params.txt'))
    net = get_net(args.model)
    
    if args.surgery == 2:
        net = to_InPlaceABN(net) 
        source = 'timm' if args.model.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)  
    elif args.surgery == 1:
        source = 'timm' if args.model.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)
    elif args.surgery == 3:
        net = remove_stride(net)
        net = add_pooling(net)
    elif args.surgery == 4:
        net = add_pooling(net)
    
    net = net.to(device)   
    
    train_dataset = TrainRetriever(
        kinds=dataset[dataset['fold'] != 0].kind.values,
        image_names=dataset[dataset['fold'] != 0].image_name.values,
        labels=dataset[dataset['fold'] != 0].label.values,
        transforms=get_train_transforms(),
        decoder=args.decoder
    )
    
    
    validation_dataset = TrainRetriever(
        kinds=dataset[dataset['fold'] == 0].kind.values,
        image_names=dataset[dataset['fold'] == 0].image_name.values,
        labels=dataset[dataset['fold'] == 0].label.values,
        transforms=get_valid_transforms(),
        decoder=args.decoder
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=TrainGlobalConfig.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )
        
    fitter = Fitter(net, train_loader, validation_loader, device, TrainGlobalConfig)
    
    if args.fp16:
        fitter.model, fitter.optimizer = amp.initialize(fitter.model, fitter.optimizer, opt_level=fitter.config.opt_level, 
                                                        loss_scale=fitter.config.loss_scale,verbosity=0)
    if args.load_checkpoint != '':
        fitter.load(args.load_checkpoint)
    fitter.fit()
    
    
if __name__ == "__main__":
    main()