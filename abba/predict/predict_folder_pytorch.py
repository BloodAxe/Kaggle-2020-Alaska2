import os
import numpy as np 
from torch import nn
import torch
import argparse
from tqdm import tqdm
from apex import amp
import pandas as pd
import sys
from torch.utils.data.sampler import SequentialSampler
sys.path.insert(1,'./')
from train.zoo.models import *
from train.zoo.surgery import *
from train.datafeeding.retriever import *
from train.tools.torch_utils import *


def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Predict Test images with TTA")
    arg = parser.add_argument
    arg('--folder', type=str, default='Test/', help='path to test folder')
    arg('--model', type=str, default='efficientnet-b4', help='model name')
    arg('--experiment', type=str, default='efficientnet_b4_NR_mish', help='specific model experiment name')
    arg('--surgery', type=int, default=2, help='modification level')
    arg('--test-time-augmentation', type=int, default=4, help='TTA level')
    arg('--checkpoint', type=str, default='' , help='path to checkpoint')
    arg('--output', type=str, default='models_predictions/', help='output folder')
    arg('--decoder', type=str, default='NR' , help='how to decode jpeg files, NR or R')
    arg('--fp16', type=int, default=1 , help='Used AMP?')
    arg('--subset', type=str, default='LB' , help='A subset of the folder? train, test or val')
    arg('--device', type=str, default='cuda:0' , help='Device')
    
    args = parser.parse_args()
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    #torch.cuda.set_device(int(args.device.split(':')[-1]))
    
    device = 'cuda:0'
    
    seed_everything(1994)
    os.makedirs(os.path.join(args.output,args.subset), exist_ok=True)
    
    net = get_net(args.model)
    if args.surgery == 2:
        net = to_InPlaceABN(net) 
        source = 'timm' if args.model.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)
    if args.surgery == 1: 
        source = 'timm' if args.model.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)
        
    
    net = net.cuda(device)     
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    if args.fp16:
        net = amp.initialize(net, None, opt_level='O1',loss_scale='dynamic',verbosity=0)
    net.eval()
    
    if args.test_time_augmentation == 4:
        TTA = dict()
        TTA[''] = lambda x: x
        TTA['rot1'] =  lambda x: np.rot90(x,1)
        TTA['rot2'] =  lambda x: np.rot90(x,2)
        TTA['rot3'] =  lambda x: np.rot90(x,3)
        TTA['fliplr'] =  lambda x: np.fliplr(x)
        TTA['fliplr_rot1'] =  lambda x: np.rot90(np.fliplr(x),1)
        TTA['fliplr_rot2'] =  lambda x: np.rot90(np.fliplr(x),2)
        TTA['fliplr_rot3'] =  lambda x: np.rot90(np.fliplr(x),3)
        
    elif args.test_time_augmentation == 1:
        TTA = dict()
        TTA[''] = lambda x: x
    
    
    IL  = os.listdir(DATA_ROOT_PATH+args.folder)
    if args.subset != 'LB':
        QFs = ['75','90', '95']
        IL = []
        for QF in QFs:
            with open('./IL_'+args.subset+'_'+QF+'.p', 'rb') as handle:
                IL.extend(pickle.load(handle))
                
    test_retriever = TestRetriever(IL, DATA_ROOT_PATH+args.folder, decoder=args.decoder)
    
    test_loader = torch.utils.data.DataLoader(
        test_retriever, 
        batch_size=10,
        shuffle=False,
        num_workers=4,
        drop_last=False)
    
    pred_dataframe = pd.DataFrame(columns=['NAME', args.experiment+'_pc', args.experiment+'_pjm', 
                                           args.experiment+'_pjuni', args.experiment+'_puerd'])
    
    pred_dataframe['NAME'] = IL
    pred_dataframe[[args.experiment+'_pc', args.experiment+'_pjm', args.experiment+'_pjuni', args.experiment+'_puerd']] = 0.0
    
    for transform in TTA.keys():
        y_preds = []
        test_loader.dataset.func_transforms = TTA[transform]
    
        for step, (image_names, images) in enumerate(tqdm(test_loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')):
            
            y_pred = net(images.cuda(device).float())
            y_pred = nn.functional.softmax(y_pred, dim=1).float().data.cpu().numpy()
            y_preds.extend(y_pred)
            
        y_preds = np.array(y_preds)
        
        pred_dataframe[[args.experiment+'_pc', args.experiment+'_pjm', args.experiment+'_pjuni', args.experiment+'_puerd']] += y_preds
        
    pred_dataframe[[args.experiment+'_pc', args.experiment+'_pjm', args.experiment+'_pjuni', args.experiment+'_puerd']] /= len(TTA.keys())
    
    output_path = os.path.join(args.output,args.subset,args.experiment+'_probabilities_'+args.folder[:-1]+'.csv')
    pred_dataframe.to_csv(output_path)
        
    del checkpoint
    del net
    
if __name__ == "__main__":
    main()