# Basic usage
import os
import csv
import random
import string
import numpy as np
import pandas as pd
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Union
import warnings
from tqdm import tqdm


# Scikit 
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC, SVR


# Tree models
from lightgbm import LGBMClassifier, plot_importance 
from imblearn.ensemble import BalancedBaggingClassifier as BBC
from imblearn.ensemble import BalancedRandomForestClassifier
# Logger & Parser
from argparse import ArgumentParser, Namespace
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
warnings.simplefilter(action='ignore')

# Dataset path
TRAIN_CSV = "train_all.csv"
PUBLIC_CSV = "public_all.csv"
PRIVATE_CSV = "private_all.csv"
TRAIN_WLM = "wavlm_train_ms.json"
PUBLIC_WLM = "wavlm_public_ms.json"
PRIVATE_WLM = "wavlm_private_ms.json"
TRAIN_W2V = "wav2vec2_train_s3prl.json"
PUBLIC_W2V = "wav2vec2_public_s3prl.json"
PRIVATE_W2V = "wav2vec2_private_s3prl.json"
# Base estimators
lgb_param = {
            'boosting_type': 'gbdt',
            'class_weight': 'balanced',
            'objective': 'multiclass'
            }
brf_param = {
            'n_estimators': 500,
            'class_weight': 'balanced_subsample',
            }
random_states = [11, 42, 17, 419]
weights = [1/8, 1/8, 1/8, 1/8, 1/2]
lgb = LGBMClassifier(**lgb_param)
brf = BalancedRandomForestClassifier(**brf_param)
# Ensembles
models = [BBC(estimator=brf, random_state=s) for s in random_states] + [BBC(estimator=lgb, random_state=42)]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default='./Dataset', help="Path to dataset."
    )
    parser.add_argument(
        "--output_path", type=str, default='./Result', help="Path to store the generation."
    )
    args = parser.parse_args()
    return args
def dimension_reduction(train, public, private, method='PCA', prefix='w2v', n_comp=20, plot=True):
    with open(train, 'r') as f:
        data = json.load(f)
    with open(public, 'r') as f:
        public = json.load(f)
    with open(private, 'r') as f:
        private = json.load(f)
    name = list(data.keys())
    data = np.array(list(data.values()))
    public_name = list(public.keys())
    public = np.array(list(public.values()))
    private_name = list(private.keys())
    private = np.array(list(private.values()))
    if method == 'PCA':
        ## PCA
        pca = PCA(n_components=n_comp)
        latent_vec = pca.fit_transform(np.vstack([data, public]))[:1000]
        # latent_vec = pca.fit_transform(data)
        public_latent = pca.transform(public)
        private_latent = pca.transform(private)
    else:
        ## TSNE
        tsne = TSNE(n_components=n_comp, verbose=0, method='exact', perplexity=15, n_iter=250)
        latent_vec = tsne.fit_transform(data)
        public_latent = tsne.fit_transform(public)
        private_latent = tsne.fit_transform(private)
    
    if plot:
        sns.barplot(x=np.arange(n_comp), y=np.cumsum(pca.explained_variance_ratio_))
    cols = [prefix + str(i) for i in range(n_comp)]
    df_train = pd.DataFrame(latent_vec, index=name, columns=cols)
    df_public = pd.DataFrame(public_latent, index=public_name, columns=cols)
    df_private = pd.DataFrame(private_latent, index=private_name, columns=cols)
    return df_train, df_public, df_private
def voting(models, train, test, weight):
    df, label = train
    n = len(models)
    y = 0
    logging.info("Ensemble2: Generate Prediction")
    for i, m in enumerate(tqdm(models)):
        m.fit(df, label)
        y+=m.predict_proba(test)*weight[i]
    return np.argmax(y, axis=1)+1, y


if __name__ == '__main__':
    ## Parsing Arguments
    args = parse_args()
    outpath = args.output_path
    os.makedirs(outpath, exist_ok=True)
    TRAIN_CSV = os.path.join(args.data_path, TRAIN_CSV)
    PUBLIC_CSV = os.path.join(args.data_path, PUBLIC_CSV)
    PRIVATE_CSV = os.path.join(args.data_path, PRIVATE_CSV)
    TRAIN_WLM = os.path.join(args.data_path, TRAIN_WLM)
    PUBLIC_WLM = os.path.join(args.data_path, PUBLIC_WLM)
    PRIVATE_WLM = os.path.join(args.data_path, PRIVATE_WLM)
    TRAIN_W2V = os.path.join(args.data_path, TRAIN_W2V)
    PUBLIC_W2V = os.path.join(args.data_path, PUBLIC_W2V)
    PRIVATE_W2V = os.path.join(args.data_path, PRIVATE_W2V)
    ## Dimension Reduction
    wlm_train, wlm_public, wlm_private = dimension_reduction(TRAIN_WLM, PUBLIC_WLM, PRIVATE_WLM, method="PCA", prefix='wlm', n_comp=9, plot=False)
    w2v_train, w2v_public, w2v_private = dimension_reduction(TRAIN_W2V, PUBLIC_W2V, PRIVATE_W2V, method="PCA", prefix='w2v', n_comp=1, plot=False)
    ## Tabular Data
    df_meta = pd.read_csv(TRAIN_CSV, index_col=0)
    df_meta_public = pd.read_csv(PUBLIC_CSV, index_col=0)
    df_meta_private = pd.read_csv(PRIVATE_CSV, index_col=0)
    ## Fill nan
    df_meta.fillna(0, inplace=True)
    df_meta_public.fillna(0, inplace=True)
    df_meta_private.fillna(0, inplace=True)
    label = df_meta['Disease category']-1
    df_meta.drop('Disease category', inplace=True, axis=1)
    ## 
    df = pd.concat([df_meta, wlm_train, w2v_train], axis=1)
    df_public = pd.concat([df_meta_public, wlm_public, w2v_public], axis=1)
    df_private = pd.concat([df_meta_private, wlm_private, w2v_private], axis=1)
    df_test = pd.concat([df_public, df_private])
    idx = df_test.index
    ##
    y_pred, y_prob = voting(models=models,
                            train=[df, label],
                            test=df_test,
                            weight=weights)
    outpath = os.path.join(outpath, 'ensemble2_proba.csv')
    pd.DataFrame(y_prob, index=idx).to_csv(outpath)