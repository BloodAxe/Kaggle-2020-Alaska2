import pandas as pd
from datetime import date
from functools import reduce
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import argparse
import pickle
import os


def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Group all zoo predictions")
    arg = parser.add_argument
    arg('--zoo-file', type=str, default='models_predictions/', help='output folder')
    arg('--catboost-file', type=str, default='models_predictions/', help='output folder')
    arg('--version', type=str, default='v26', help='output folder')
    
    args = parser.parse_args()
    today = date.today()
    d = today.strftime('%m%d')
    
    df_features = pd.read_csv(args.zoo_file, index_col=0) 
    model = CatBoostClassifier()
    model = model.load_model(args.catboost_file)
    
    df_features = df_features[['NAME','QF', 'DCTR', 'JRM', 
                               'SRNet_pc', 'SRNet_pjm', 'SRNet_pjuni', 'SRNet_puerd',
                               'efficientnet_b2_pc', 'efficientnet_b2_pjm', 'efficientnet_b2_pjuni', 'efficientnet_b2_puerd',
                               'mixnet_S_pc', 'efficientnet_b4_NR_mish_pc', 'mixnet_xL_NR_mish_pc', 
                               'efficientnet_b5_NR_mish_pc']]
    df_features['QF'] = df_features['QF'].astype('int').astype('str')
    X  = df_features.values[:,1:]
    
    scores = model.predict_proba(X)[:,1]
    sub = pd.DataFrame({"Id":df_features.NAME, "Label":scores})
    
    with open('models_predictions/LB/out_of_bounds_Test.p', 'rb') as handle:
        oor = pickle.load(handle)
        
    for im_name in oor:
        sub.loc[sub.Id==im_name, 'Label'] = 1.01
        
    sub.to_csv('submissions/submission_'+args.version+'.csv', index=False)
    plt.figure(figsize=(7,5))
    plt.hist(sub.Label, bins=50)
    plt.savefig('submissions/submission_'+args.version+'_histogram.png')
    

if __name__ == "__main__":
    main()