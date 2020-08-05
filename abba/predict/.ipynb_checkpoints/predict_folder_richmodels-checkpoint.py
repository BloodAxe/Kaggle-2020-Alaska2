import os
import numpy as np 
import argparse
from tqdm import tqdm
#from pqdm.processes import pqdm
#from joblib import Parallel, delayed
#from joblib.externals.loky import set_loky_pickler
#from joblib import wrap_non_picklable_objects
#from joblib import parallel_backend
import pandas as pd
import sys
import pickle
import jpegio as jio
from oct2py import octave
octave.addpath('rich_models/')

def main():
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Predict Test images using rich models")
    arg = parser.add_argument
    arg('--folder', type=str, default='Test/', help='path to test folder')
    arg('--experiment', type=str, default='DCTR', help='specific model experiment name')
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
            
    if args.experiment == 'DCTR':
        f = octave.DCTR
    elif args.experiment == 'JRM':
        f = octave.JRM
            
    votes = []
    for im_name in tqdm(IL, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'): 
        tmp = jio.read(os.path.join(folder, im_name))
        feature = f(tmp.coef_arrays[0], tmp.quant_tables[0])
        fld_ensemble_prediction = octave.ensemble_testing(feature, args.checkpoint)
        votes.append(fld_ensemble_prediction['votes'])
        
    pred_dataframe = pd.DataFrame(columns=['NAME', args.experiment])
    
    pred_dataframe['NAME'] = IL
    pred_dataframe[args.experiment] = votes
    
    output_path = os.path.join(args.output, args.subset, 'QF'+str(args.quality_factor)+'_'+args.experiment+'_votes_'+args.folder[:-1]+'.csv')
    pred_dataframe.to_csv(output_path)
    
if __name__ == "__main__":
    main()