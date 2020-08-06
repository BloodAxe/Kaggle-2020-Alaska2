import numpy as np
import pandas as pd
import pickle
import glob
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, GroupKFold, ShuffleSplit
import sys
sys.path.insert(1,'./')
from train.tools.kaggle_tools import wauc

import argparse
from tqdm import tqdm
import os

class GroupShuffleSplit:
    def __init__(self, names_groups, test_size=1000, n_splits=3):
        self.n_splits = n_splits
        self.names_groups = names_groups
        self.unique_names = list(self.names_groups.keys())
        self.test_size = test_size    
    def split(self, X, y, groups=None):
        for _ in range(self.n_splits):
            np.random.shuffle(self.unique_names)
            train_names = self.unique_names[self.test_size:]
            train_idx = np.hstack([self.names_groups[i] for i in train_names]) if len(train_names)>0 else []
            test_names = self.unique_names[:self.test_size]
            test_idx = np.hstack([self.names_groups[i] for i in test_names])
            yield train_idx, test_idx 
    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

def main():
    # This uses SKOPT to fit hyperparameters of catboost
    # Adapted from https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html#sphx-glr-auto-examples-sklearn-gridsearchcv-replacement-py
    
    parser = argparse.ArgumentParser("Fit Catboost for 2nd level stacking")
    arg = parser.add_argument
    arg('--zoo-files-dir', type=str, default='models_predictions/', help='path to zoo file')
    arg('--zoo-id', type=str, default='0805', help='zoo id (date in mmdd format)')
    arg('--train-dir', type=str, default='weights/catboost/', help='path to catboost train dir')
    arg('--n-splits', type=int, default=10 , help='num CV splits')
    arg('--n-iter', type=int, default=60, help='num Bayes opt iters') 

    args = parser.parse_args()
    
    os.makedirs(args.train_dir, exist_ok=True)
    
    probabilities_zoo_holdout = pd.DataFrame()
    
    for fold in ['val', 'test']:
        folder = os.path.join(args.zoo_files_dir, fold)
        files = glob.glob(os.path.join(folder, '*'+args.zoo_id+'.csv'))
        for c in ['Cover','UERD','JMiPOD','JUNIWARD']:
            file = [f for f in files if c in f][0]
            df = pd.read_csv(file, index_col=0)
            df['CLASS'] = c
            df['FOLD'] = fold
            probabilities_zoo_holdout = probabilities_zoo_holdout.append(df, ignore_index=True)
    
    # probabilities_zoo_holdout = pd.read_csv(args.zoo_file, index_col=0)
    
    FEATURES = ['NAME', 'QF', 'DCTR', 'JRM', 
                'SRNet_pc', 'SRNet_pjm', 'SRNet_pjuni', 'SRNet_puerd',
                'efficientnet_b2_pc', 'efficientnet_b2_pjm', 'efficientnet_b2_pjuni', 'efficientnet_b2_puerd',
                'mixnet_S_pc', 'efficientnet_b4_NR_mish_pc', 'mixnet_xL_NR_mish_pc',
                'efficientnet_b5_NR_mish_pc']
    
    with open(os.path.join(args.train_dir, 'FEATURES'+'.p'), 'wb') as handle:
        pickle.dump(FEATURES, handle)
    
    
    df_features = probabilities_zoo_holdout[probabilities_zoo_holdout.FOLD == 'val']
    labels = ~np.array(df_features.CLASS == 'Cover') + 0
    df_features = df_features[FEATURES]
    names = df_features.NAME
    names_groups = names.groupby(names).groups
    
    def scoring(estimator,X,y):
        scores = 1-estimator.predict_proba(X)[:,0]
        return wauc(y>0, scores)
    
    bayes_cv_tuner = BayesSearchCV(
        
         estimator = CatBoostClassifier(
             objective = 'Logloss',
             cat_features = [0],
             iterations=150,
             scale_pos_weight=1/3.0,
             train_dir=args.train_dir,
             silent=True),
            
        search_spaces = {
            'learning_rate': (0.0001, 1.0, 'log-uniform'),
            'depth': (2, 13),
            'colsample_bylevel': (0.5, 1.0, 'uniform'),
            'l2_leaf_reg': (1.0, 50.0, 'uniform'),
            
        },    
        scoring = scoring,
        cv = GroupShuffleSplit(
            n_splits = args.n_splits,
            names_groups = names_groups
        ),
        n_jobs = args.n_splits,
        n_iter = args.n_iter,   
        verbose = 0,
        refit = True,
        random_state = None
    )
    
    def on_step(optim_result):
        all_models = bayes_cv_tuner.cv_results_  
        best_params = bayes_cv_tuner.best_params_
        print('Model #{}\nBest wAUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))
        
        
    df_features['QF'] = df_features['QF'].astype('int').astype('str')
    X = df_features.values[:,1:]
    
    result = bayes_cv_tuner.fit(X, labels, callback=on_step)
    
    means = np.array(result.cv_results_['mean_test_score'])
    means_cv = [np.array(result.cv_results_['split'+str(k)+'_test_score']) for k in range(bayes_cv_tuner.cv.n_splits)]
    stds = np.array(result.cv_results_['std_test_score'])
    ranks = np.array(result.cv_results_['rank_test_score'])
    plt.figure(figsize=(7,5))
    for k in range(bayes_cv_tuner.cv.n_splits):
        plt.plot(means_cv[k], alpha=0.5)
        
    plt.scatter(np.where(ranks==1)[0],means[ranks==1], label='optimum')
    
    plt.ylabel(str(bayes_cv_tuner.cv.n_splits)+' folds CV wauc')
    plt.xlabel('Hyper parameter optimization steps')
    plt.ylim([0.93,0.95])
    plt.legend()
    plt.savefig(args.train_dir + 'skopt_iterations.png')
    
    
    df_features = probabilities_zoo_holdout[probabilities_zoo_holdout.FOLD == 'test']
    labels = ~np.array(df_features.CLASS == 'Cover') + 0
    df_features = df_features[FEATURES]
    df_features['QF'] = df_features['QF'].astype('int').astype('str')
    X = df_features.values[:,1:]
    scores = result.best_estimator_.predict_proba(X)[:,1]
    TST_wauc = wauc(labels, scores)
    TST_wauc = np.round(TST_wauc, 5)
    result.best_estimator_.save_model(args.train_dir + 'best_catboost_TST_'+str(TST_wauc)+'.cmb')
    

if __name__ == "__main__":
    main()