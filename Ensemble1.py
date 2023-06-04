import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd

from copy import deepcopy
from collections import OrderedDict
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from sklearn.decomposition import PCA
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier


BRF_PARAMS = {
    'n_estimators': 500,
    'class_weight': 'balanced_subsample',
}

LGBM_PARAMS = {
    'boosting_type': 'gbdt',
    'class_weight': 'balanced',
    'objective': 'multiclass'
}

TAB_PARAMS = {
    'N_ensemble_configurations': 100
}


def hidden_message(func):
    def wrap(path, *args):
        sys.stdout = open(os.devnull, 'w')
        func(path, *args)
        sys.stdout.close()
    return wrap

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

DATA_FOLDER = "dataset"
TRAIN_TABULAR_FILE = "tabular_train.csv"
TRAIN_WAVLM_FILE = "wavlm_train_ms.json"
TRAIN_WAV2VEC_FILE = "wav2vec2_train_s3prl.json"
PUBLIC_WAVLM_FILE = "wavlm_public_ms.json"
PUBLIC_TABULAR_FILE = "tabular_public.csv"
PUBLIC_WAV2VEC_FILE = "wav2vec2_public_s3prl.json"
PRIVATE_WAVLM_FILE = "wavlm_private_ms.json"
PRIVATE_TABULAR_FILE = "tabular_private.csv"
PRIVATE_WAV2VEC_FILE = "wav2vec2_private_s3prl.json"
OUTPUT_FOLDER = "prediction"
OUTPUT_FILE = "submissions.csv"

TRAIN_TABULAR_PATH = os.path.join(DATA_FOLDER, TRAIN_TABULAR_FILE)
TRAIN_WAVLM_PATH = os.path.join(DATA_FOLDER, TRAIN_WAVLM_FILE)
TRAIN_WAV2VEC_PATH = os.path.join(DATA_FOLDER, TRAIN_WAV2VEC_FILE)
PUBLIC_TABULAR_PATH = os.path.join(DATA_FOLDER, PUBLIC_TABULAR_FILE)
PUBLIC_WAVLM_PATH = os.path.join(DATA_FOLDER, PUBLIC_WAVLM_FILE)
PUBLIC_WAV2VEC_PATH = os.path.join(DATA_FOLDER, PUBLIC_WAV2VEC_FILE)
PRIVATE_WAVLM_PATH = os.path.join(DATA_FOLDER, PRIVATE_WAVLM_FILE)
PRIVATE_TABULAR_PATH = os.path.join(DATA_FOLDER, PRIVATE_TABULAR_FILE)
PRIVATE_WAV2VEC_PATH = os.path.join(DATA_FOLDER, PRIVATE_WAV2VEC_FILE)
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

# model
BRF = "brf"
LGBM = "lgbm"
TAB = "tab"
BBC = "bbc"

# data
TABULAR = "tabular"
WAVLM_ENCODE = "wavlm-encode"
WAV2VEC_ENCODE = "wav2vec-encode"


TRAIN_DATA = OrderedDict([
    (TABULAR, TRAIN_TABULAR_PATH),
    (WAVLM_ENCODE, TRAIN_WAVLM_PATH),
    (WAV2VEC_ENCODE, TRAIN_WAV2VEC_PATH),
])

PUBLIC_DATA = OrderedDict([
    (TABULAR, PUBLIC_TABULAR_PATH),
    (WAVLM_ENCODE, PUBLIC_WAVLM_PATH),
    (WAV2VEC_ENCODE, PUBLIC_WAV2VEC_PATH),
])

PRIVATE_DATA = OrderedDict([
    (TABULAR, PRIVATE_TABULAR_PATH),
    (WAVLM_ENCODE, PRIVATE_WAVLM_PATH),
    (WAV2VEC_ENCODE, PRIVATE_WAV2VEC_PATH),
])

LABEL_COLUMN = "Disease category"


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class ModelSetting():
    def __init__(self, trans, models):
        self.trans = trans
        self.models = models


def get_model_settings():
    return [
        ModelSetting(
            trans="v1",
            models=[
                OrderedDict(
                    algo='brf-bbc',
                    brf_params=BRF_PARAMS,
                    bbc_params={'random_state': 11}
                ),
                OrderedDict(
                    algo='lgbm-bbc',
                    lgbm_params=LGBM_PARAMS,
                    bbc_params={'random_state': 42}
                ),
                OrderedDict(
                    algo='tab-bbc',
                    tab_params=TAB_PARAMS,
                    bbc_params={'random_state': 42}
                ),
            ]
        ),
        ModelSetting(
            trans="v2",
            models=[
                OrderedDict(
                    algo='brf-bbc-1',
                    brf_params=BRF_PARAMS,
                    bbc_params={'random_state': 11},
                    weight=0.125
                ),
                OrderedDict(
                    algo='brf-bbc-2',
                    brf_params=BRF_PARAMS,
                    bbc_params={'random_state': 17},
                    weight=0.125
                ),
                OrderedDict(
                    algo='brf-bbc-3',
                    brf_params=BRF_PARAMS,
                    bbc_params={'random_state': 42},
                    weight=0.125
                ),
                OrderedDict(
                    algo='brf-bbc-4',
                    brf_params=BRF_PARAMS,
                    bbc_params={'random_state': 419},
                    weight=0.125
                ),
                OrderedDict(
                    algo='lgbm-bbc',
                    lgbm_params=LGBM_PARAMS,
                    bbc_params={'random_state': 42},
                    weight=0.5
                ),
            ]
        ),
    ]


class DataPipeline():
    def __init__(self, transform_list):
        self.data_type = sorted(list(set([k for d in transform_list for k in d['keys']])), key=len)
        self.transform_list = transform_list

    def concat(self, data):
        X = []
        for k in self.data_type:
            if isinstance(data[k], pd.DataFrame):
                X.append(data[k].values)
            else:
                X.append(data[k])
        return np.concatenate(X, axis=1)

    def get_X_and_y(self, data):
        y = data[TABULAR][LABEL_COLUMN].values - 1
        data[TABULAR] = data[TABULAR].drop(LABEL_COLUMN, axis=1)
        X = self.concat(data)
        return X, y

    def fit_transform(self, data):
        data = deepcopy(data)
        for d in self.transform_list:
            for k in d['keys']:
                data[k] = d['transform'].fit_transform(data[k])
        X, y = self.get_X_and_y(data)
        return OrderedDict(X=X, y=y, id=data[TABULAR].index)

    def transform(self, data):
        data = deepcopy(data)
        for d in self.transform_list:
            for k in d['keys']:
                data[k] = d['transform'].transform(data[k])
        X = self.concat(data)
        return OrderedDict(X=X, y=None, id=data[TABULAR].index)


class BaseTransform():
    def fit_transform(self, data):
        raise NotImplementedError
    
    def transform(self, data):
        raise NotImplementedError


class LoadTabularData(BaseTransform):
    def __init__(self):
        super(LoadTabularData, self).__init__()

    def read_csv(self, data):
        return pd.read_csv(data, index_col=0)

    def fit_transform(self, data):
        return self.read_csv(data)

    def transform(self, data):
        return self.read_csv(data)


class LoadAudioData(BaseTransform):
    def __init__(self):
        super(LoadAudioData, self).__init__()

    def fit_transform(self, data):
        data = load_json(data).values()
        return np.array(list(data))
    
    def transform(self, data):
        data = load_json(data).values()
        return np.array(list(data))


class FillNa(BaseTransform):
    def __init__(self):
        super(FillNa, self).__init__()

    def fit_transform(self, data):
        return data.fillna(0)
    
    def transform(self, data):
        return data.fillna(0)


class EnsemblePipeline():
    def __init__(self, pipeline_dict, weights_dict):
        self.pipeline_dict = pipeline_dict
        self.weights_dict = weights_dict

    def predict_proba(self, X):
        y_pred_probas = []
        for name, pipeline in self.pipeline_dict.items():
            proba = pipeline.predict_proba(X)
            y_pred_probas.append(self.weights_dict[name] * proba)
        return np.sum(y_pred_probas, axis=0)

    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        y_pred_lab = np.argmax(y_pred_proba, axis=1) + 1  # convert to label
        return y_pred_lab


class Learner():
    def __init__(self, models):
        self.models = models
    
    @hidden_message
    def init_model(self):
        logging.info(f'Initial models')
        pipeline_dict = OrderedDict([])
        for model in self.models:
            algo = model['algo']
            if BRF in algo:
                predictor = BalancedRandomForestClassifier(**model['brf_params'])
            elif LGBM in algo:
                predictor = LGBMClassifier(**model['lgbm_params'])
            elif TAB in algo:
                predictor = TabPFNClassifier(**model['tab_params'])
            if BBC in algo:
                predictor = BalancedBaggingClassifier(
                    estimator=predictor,
                    **model['bbc_params']
                )
            pipeline_dict[algo] = predictor

        weights_dict = {model['algo']: model['weight'] for model in self.models} \
                       if 'weight' in self.models[0].keys() else \
                       {p: 1 / len(pipeline_dict) for p in pipeline_dict.keys()}
        self.ensem_pipeline = EnsemblePipeline(pipeline_dict, weights_dict)
        return

    @hidden_message
    def train(self, X, y):
        logging.info(f'Training')
        for name, predictor in self.ensem_pipeline.pipeline_dict.items():
            predictor.fit(X, y)
        return

    def predict(self, X):
        logging.info(f'Prediction')
        y_pred_lab = self.ensem_pipeline.predict(X)
        return y_pred_lab
    
    def predict_proba(self, X):
        logging.info(f'Prediction')
        y_pred_proba = self.ensem_pipeline.predict_proba(X)
        return y_pred_proba



def save_pred(path, pred_df):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pred_df.to_csv(path)
    return


TRANS_LIST_v1 = [
    OrderedDict(keys=[TABULAR], transform=LoadTabularData()),
    OrderedDict(keys=[WAVLM_ENCODE], transform=LoadAudioData()),
    OrderedDict(keys=[TABULAR], transform=FillNa()),
    OrderedDict(keys=[WAVLM_ENCODE], transform=PCA(n_components=9)),
]

TRANS_LIST_v2 = [
    OrderedDict(keys=[TABULAR], transform=LoadTabularData()),
    OrderedDict(keys=[WAVLM_ENCODE, WAV2VEC_ENCODE], transform=LoadAudioData()),
    OrderedDict(keys=[TABULAR], transform=FillNa()),
    OrderedDict(keys=[WAVLM_ENCODE], transform=PCA(n_components=9)),
    OrderedDict(keys=[WAV2VEC_ENCODE], transform=PCA(n_components=1)),
]

TRANS_LIST = {
    "v1": TRANS_LIST_v1,
    "v2": TRANS_LIST_v2,
}


if __name__ == "__main__":
    model_settings = get_model_settings()
    for setting in model_settings:
        trans_list = TRANS_LIST[setting.trans]
        models = setting.models

        data_pipeline = DataPipeline(trans_list)
        train_data = data_pipeline.fit_transform(TRAIN_DATA)
        public_data = data_pipeline.transform(PUBLIC_DATA)
        private_data = data_pipeline.transform(PRIVATE_DATA)

        learner = Learner(models=models)
        learner.init_model()
        learner.train(train_data['X'], train_data['y'])

        public_pred_proba = learner.predict_proba(public_data['X'])
        private_pred_proba = learner.predict_proba(private_data['X'])

        pred_df = pd.concat([
            pd.DataFrame(public_pred_proba, index=public_data['id']),
            pd.DataFrame(private_pred_proba, index=private_data['id']),
        ])

        if OUTPUT_PATH is not None:
            save_name = f"trans={setting.trans}_model={'_'.join([m['algo'] for m in models])}.csv"
            save_pred(os.path.join(OUTPUT_FOLDER, save_name), pred_df)
