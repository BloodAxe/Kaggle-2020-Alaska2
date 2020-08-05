import pickle
import os 
import sys
import argparse
from tqdm import tqdm
sys.path.insert(1,'./')
from train.zoo.out_of_bounds import *


def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Detect out of bounds images from folder")
    arg = parser.add_argument
    arg('--folder', type=str, default='Test/', help='path to test folder')
    arg('--output', type=str, default='models_predictions/', help='output folder')
    arg('--subset', type=str, default='LB' , help='A subset of the folder? train, test or val')
    
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output, args.subset), exist_ok=True)
    
    folder = os.path.join(DATA_ROOT_PATH, args.folder)
    IL = os.listdir(folder)
    
    if args.subset != 'LB':
        QFs = ['75','90', '95']
        IL = []
        for QF in QFs:
            with open('./IL_'+args.subset+'_'+QF+'.p', 'rb') as handle:
                IL.extend(pickle.load(handle))
    
    
    bounds = get_DCT_bounds()
    out_of_bounds = []
    
    for name in tqdm(IL, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        outlier_name = is_outlier(name, folder, bounds['max'], bounds['min'])
        if outlier_name is not None:
            out_of_bounds.append(outlier_name)
        
        
    print('Found ',len(out_of_bounds), ' OOB images in folder')
        
    with open(os.path.join(args.output, args.subset,'out_of_bounds_'+args.folder[:-1]+'.p'), 'wb') as handle:
        pickle.dump(out_of_bounds, handle)
    

if __name__ == "__main__":
    main()