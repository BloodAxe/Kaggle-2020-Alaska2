import os
import numpy as np 
import argparse
from tqdm import tqdm
import pandas as pd
import sys
import pickle
sys.path.insert(1,'./')
from train.zoo.models import *
from train.tools.tf_utils import *
from train.tools.jpeg_utils import *

def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Predict Test images with TTA D4")
    arg = parser.add_argument
    arg('--folder', type=str, default='Test/', help='path to test folder')
    arg('--experiment', type=str, default='SRNet', help='specific model experiment name')
    arg('--checkpoint', type=str, default='' , help='path to checkpoint')
    arg('--quality-factor', type=int, default=75 , help='quality factor')
    arg('--output', type=str, default='models_predictions/', help='output folder')
    arg('--subset', type=str, default='LB' , help='A subset of the folder? train, test or val')
    
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output, args.subset), exist_ok=True)
    folder = os.path.join(DATA_ROOT_PATH, args.folder)
    
    if args.subset == 'LB':
        names = os.listdir(folder)
        test_qf_dicts_path = os.path.join(DATA_ROOT_PATH, 'Test_qf_dicts.p')
        if not os.path.exists(test_qf_dicts_path):
            (names_qf, qf_names) = get_qf_dicts(folder, names)
            with open(test_qf_dicts_path, 'wb') as handle:
                pickle.dump((names_qf, qf_names), handle)
        else:
            with open(test_qf_dicts_path, 'rb') as handle:
                (names_qf, qf_names) = pickle.load(handle)
                
        IL = qf_names[args.quality_factor]
        
    else:
        with open('./IL_'+args.subset+'_'+str(args.quality_factor)+'.p', 'rb') as handle:
            IL = pickle.load(handle)
                
    TTA = dict()
    TTA[''] = lambda x: x
    TTA['rot1'] =  lambda x: np.rot90(x,1)
    TTA['rot2'] =  lambda x: np.rot90(x,2)
    TTA['rot3'] =  lambda x: np.rot90(x,3)
    TTA['fliplr'] =  lambda x: np.fliplr(x)
    TTA['fliplr_rot1'] =  lambda x: np.rot90(np.fliplr(x),1)
    TTA['fliplr_rot2'] =  lambda x: np.rot90(np.fliplr(x),2)
    TTA['fliplr_rot3'] =  lambda x: np.rot90(np.fliplr(x),3)
    
    probabilities, file_names = test_predict(SR_net_model_eff, folder, IL, args.checkpoint, 4, TTA)
        
    pred_dataframe = pd.DataFrame(columns=['NAME', args.experiment+'_pc', args.experiment+'_pjm', 
                                           args.experiment+'_pjuni', args.experiment+'_puerd'])
    
    pred_dataframe['NAME'] = file_names
    pred_dataframe[[args.experiment+'_pc', args.experiment+'_pjm', args.experiment+'_pjuni', args.experiment+'_puerd']] = probabilities
    
    output_path = os.path.join(args.output, args.subset ,'QF'+str(args.quality_factor)+'_'+args.experiment+'_probabilities_'+args.folder[:-1]+'.csv')
    pred_dataframe.to_csv(output_path)
    
if __name__ == "__main__":
    main()