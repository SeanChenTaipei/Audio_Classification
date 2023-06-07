import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd

from collections import OrderedDict

from transform import TRANS
from learner import Learner
from dataset import Dataset
from pipeline import DataPipeline

from model_params import BRF_PARAMS, LGBM_PARAMS, TAB_PARAMS
from constant import TRAIN_DATA, PUBLIC_DATA, PRIVATE_DATA, LABEL_COLUMN
from constant import OUTPUT_FOLDER
from constant import TABULAR, WAVLM_ENCODE, WAV2VEC_ENCODE
from constant import BRF, LGBM, TAB, BBC



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)



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
                # OrderedDict(
                #     algo='lgbm-bbc',
                #     lgbm_params=LGBM_PARAMS,
                #     bbc_params={'random_state': 42}
                # ),
                # OrderedDict(
                #     algo='tab-bbc',
                #     tab_params=TAB_PARAMS,
                #     bbc_params={'random_state': 42}
                # ),
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
                # OrderedDict(
                #     algo='brf-bbc-2',
                #     brf_params=BRF_PARAMS,
                #     bbc_params={'random_state': 17},
                #     weight=0.125
                # ),
                # OrderedDict(
                #     algo='brf-bbc-3',
                #     brf_params=BRF_PARAMS,
                #     bbc_params={'random_state': 42},
                #     weight=0.125
                # ),
                # OrderedDict(
                #     algo='brf-bbc-4',
                #     brf_params=BRF_PARAMS,
                #     bbc_params={'random_state': 419},
                #     weight=0.125
                # ),
                # OrderedDict(
                #     algo='lgbm-bbc',
                #     lgbm_params=LGBM_PARAMS,
                #     bbc_params={'random_state': 42},
                #     weight=0.5
                # ),
            ]
        ),
    ]

def save_pred(path, pred_df):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pred_df.to_csv(path)
    return


if __name__ == "__main__":
    model_settings = get_model_settings()
    for setting in model_settings:
        data_pipeline = DataPipeline(TRANS[setting.trans])
        train_data = data_pipeline.fit_transform(TRAIN_DATA)
        public_data = data_pipeline.transform(PUBLIC_DATA)
        private_data = data_pipeline.transform(PRIVATE_DATA)
        test_data = Dataset.merge(public_data, private_data)

        learner = Learner(models=setting.models)
        learner.init_model()
        learner.train(train_data.X, train_data.y)

        test_pred_proba = learner.predict_proba(test_data.X)
        test_pred_df = pd.DataFrame(test_pred_proba, index=test_data.ids)

        if OUTPUT_FOLDER is not None:
            save_name = f"trans={setting.trans}_model={'_'.join([m['algo'] for m in setting.models])}.csv"
            save_pred(os.path.join(OUTPUT_FOLDER, save_name), test_pred_df)
