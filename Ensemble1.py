import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from argparse import ArgumentParser, Namespace
import warnings
import logging
import os
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_wav",
        type=str,
        default="./Dataset/wavlm_training_ms.json",
        help="The path of directory of the training audio datas.")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="./Dataset/train_all.csv",
        help="The path of the training csv."
    )
    parser.add_argument(
        "--public_wav",
        type=str,
        default="./Dataset/wavlm_public_ms.json",
        help="The path ofdirectory of the training audio datas."
    )
    parser.add_argument(
        "--public_csv",
        type=str,
        default="./Dataset/public_all.csv",
        help="The path of the public csv."
    )
    parser.add_argument(
        "--private_wav",
        type=str,
        default="./Dataset/wavlm_private_ms.json",
        help="The  path ofdirectory of the training audio datas."
    )
    parser.add_argument(
        "--private_csv",
        type=str,
        default="./Dataset/private_all.csv",
        help="The path of the private csv."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./Result',
        help="Path to store the generation."
    )
    args = parser.parse_args()
    return args


def load_tabular(root):
    tabular_df = pd.read_csv(root, index_col=0)
    tabular_df["Disease category"] = tabular_df["Disease category"] - 1
    tabular_df = tabular_df.fillna(0)
    return tabular_df


def load_audio(root, model=None):
    with open(root, 'r') as f:
        data = json.load(f)
    new_X = np.array(list(data.values()))
    new_X = pd.DataFrame(new_X, index=list(data))
    if model is not None:
        new_X = model.fit_transform(new_X)
        new_X = pd.DataFrame(new_X, index=list(data))
        return new_X, model
    else:
        new_X = pd.DataFrame(new_X, index=list(data))
        return new_X


def load_public_private(audio_root, tabular_root, model):
    audio = load_audio(audio_root)
    audio = model.transform(audio)
    tabular = pd.read_csv(tabular_root, index_col=0)
    X = np.concatenate((audio, tabular.values), axis=1)
    X = pd.DataFrame(X, index=tabular.index)
    X = X.fillna(0)
    return X


def train(X, y, X_public, X_private, models, output_path=None, weights=None):
    weights = [0.5] * len(models) if not weights else weights
    y_prob_public = np.zeros((500, 5))
    y_prob_private = np.zeros((500, 5))
    for model, weight in zip(models, weights):
        with HiddenPrints():
            model.fit(X, y)
        y_prob_public += model.predict_proba(X_public) * weight
        y_prob_private += model.predict_proba(X_private) * weight

    y_pred_public = np.argmax(y_prob_public, axis=1)
    y_pred_private = np.argmax(y_prob_private, axis=1)

    ans_public = y_pred_public + 1
    ans_df_public = pd.DataFrame(ans_public, index=X_public.index)

    ans_private = y_pred_private + 1
    ans_df_private = pd.DataFrame(ans_private, index=X_private.index)

    final_df = pd.concat((ans_df_public, ans_df_private))

    if output_path is not None:
        y_prob_public = pd.DataFrame(y_prob_public, index=X_public.index)
        y_prob_private = pd.DataFrame(y_prob_private, index=X_private.index)
        y_prob = pd.concat((y_prob_public, y_prob_private))
        y_prob.to_csv(os.path.join(output_path, "ensemble1_proba.csv"))

    return final_df, y_prob


if __name__ == "__main__":
    args = parse_args()
    outpath = args.output_path
    os.makedirs(outpath, exist_ok=True)

    # load traing data
    logging.info(f'Loding training Data')
    tabular_df = load_tabular(args.train_csv)
    tabular_X = tabular_df.drop("Disease category", axis=1)
    y = tabular_df["Disease category"]
    pca = PCA(n_components=9)
    audio_df, pca = load_audio(args.train_wav, model=pca)
    X = np.concatenate((audio_df.values, tabular_X.values), axis=1)
    X = pd.DataFrame(X, index=audio_df.index)

    # load public and private data
    logging.info(f'Loding Public and Private Data')
    X_public = load_public_private(args.public_wav, args.public_csv, pca)
    X_private = load_public_private(args.private_wav, args.private_csv, pca)

    # construct models
    logging.info(f'Constructing Models')
    brf = BalancedRandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample")
    bbc_brf = BalancedBaggingClassifier(
        estimator=brf,
        random_state=11,
    )
    lgb = LGBMClassifier()
    bbc_lgb = BalancedBaggingClassifier(
        estimator=lgb,
        random_state=42
    )
    with HiddenPrints():
        tab = TabPFNClassifier(N_ensemble_configurations=100)
    bbc_tab = BalancedBaggingClassifier(
        estimator=tab,
        random_state=42
    )

    # train model
    logging.info(f'Training Model')
    weights = [1 / 3, 1 / 3, 1 / 3]
    final_df, y_prob = train(
        X,
        y,
        X_public,
        X_private,
        [bbc_brf, bbc_lgb, bbc_tab],
        weights=weights,
        output_path=args.output_path
    )
