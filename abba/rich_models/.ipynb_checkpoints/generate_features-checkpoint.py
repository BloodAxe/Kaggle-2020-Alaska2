import argparse
import sys
import numpy as np
import os
import pickle
import jpegio as jio
from oct2py import octave
from tqdm import tqdm
octave.addpath('rich_models/')
#sys.path.insert(1,'./')


def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Generates rich models features")
    arg = parser.add_argument
    arg('--model', type=str, default='DCTR', help='model name')
    arg('--folder', type=str, default='Cover/', help='model name')
    arg('--subset', type=str, default='train', help='split')
    arg('--output', type=str, default='train/features/', help='model name')
    arg('--quality-factor', type=int, default=75, help='quality factor')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    if args.model == 'DCTR':
        f = octave.DCTR
    elif args.model == 'JRM':
        f = octave.JRM
        
    with open('./IL_'+args.subset+'_'+str(args.quality_factor)+'.p', 'rb') as handle:
        IL = pickle.load(handle)
                
    im_name = IL[0]
    tmp = jio.read(os.path.join(DATA_ROOT_PATH+'Cover', im_name))
    feature = f(tmp.coef_arrays[0], tmp.quant_tables[0])
    dim = feature.shape[1]
    
    features = np.zeros((len(IL), dim))
    for i, im_name in enumerate(tqdm(IL, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')): 
        tmp = jio.read(os.path.join(DATA_ROOT_PATH+args.folder, im_name))
        features[i,:] = f(tmp.coef_arrays[0], tmp.quant_tables[0])    
        
    np.save(os.path.join(args.output,'QF'+str(args.quality_factor)+'_'+args.model+'_'+args.subset+'_features_'+args.folder[:-1]), features)

if __name__ == "__main__":
    main()