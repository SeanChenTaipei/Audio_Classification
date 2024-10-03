import logging
import pickle
import os
from copy import deepcopy

import pandas as pd
import numpy as np

from src.constant import TABULAR, LABEL_COLUMN
from src.dataset import Dataset
from src.estimator import Estimator
from src.transform import TRANS
from src.utils import hidden_message


class DataPipeline:
    def __init__(self, trans_name):
        self.trans_name = trans_name
        self.transforms = TRANS[trans_name]

    def concat(self, data):
        X = []
        data_type = sorted(list(set([t[0] for t in self.transforms])), key=len)

        for k in data_type:
            if isinstance(data[k], pd.DataFrame):
                X.append(data[k].values)
            else:
                X.append(data[k])

        return np.concatenate(X, axis=1)

    def get_X_and_y(self, data):
        y = data[TABULAR].pop(LABEL_COLUMN).values - 1
        X = self.concat(data)
        return np.array(X), np.array(y)

    def fit_transform(self, data):
        data = deepcopy(data)
        for name, func in self.transforms:
            data[name] = func.fit_transform(data[name])
        X, y = self.get_X_and_y(data)
        return Dataset(X=X, y=y, ids=data[TABULAR].index.to_numpy())

    def transform(self, data):
        data = deepcopy(data)
        for name, func in self.transforms:
            data[name] = func.fit_transform(data[name])
        X = self.concat(data)
        return Dataset(X=X, ids=data[TABULAR].index)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{self.trans_name}.pkl"), "wb") as f:
            pickle.dump(self.transforms, f)
            f.close()

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            transforms = pickle.load(f)
            f.close()
        pipe = DataPipeline.__new__(DataPipeline)
        pipe.transforms = transforms
        return pipe


class Pipeline:
    def __init__(self, model_setting, logger=logging):
        self.data_pipeline = DataPipeline(
            trans_name=model_setting.trans
        )
        self.estimator = Estimator(
            algo=model_setting.algo,
            params=model_setting.params
        )
        self.logger = logger

    def train(self, data):
        data = self.data_pipeline.fit_transform(data)
        self.estimator.fit(data.X, data.y)

    def predict_proba(self, data):
        data = self.data_pipeline.transform(data)
        pred_proba = self.estimator.predict_proba(data.X)
        return pred_proba

    def predict(self, data):
        data = self.data_pipeline.transform(data)
        pred_lab = self.estimator.predict(data.X)
        return pred_lab

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.data_pipeline.save(path)
        self.estimator.save(path)

    @staticmethod
    def load(path, model_setting, logger=logging):
        data_pipeline = DataPipeline.load(os.path.join(path, f"{model_setting.trans}.pkl"))
        estimator = Estimator.load(os.path.join(path, f"{model_setting.algo}.pkl"))
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.data_pipeline = data_pipeline
        pipeline.estimator = estimator
        pipeline.logger = logger
        return pipeline


class EnsemblePipeline:
    def __init__(self, model_settings, logger=logging):
        self.pipelines = {
            model.name: Pipeline(model)
            for model in model_settings
        }
        self.weights = {
            model.name: model.weight
            for model in model_settings
        }
        self.logger = logger

    @hidden_message
    def train(self, data):
        self.logger.info("Training Processing")
        for pipeline in self.pipelines.values():
            pipeline.train(data)

    def predict_proba(self, data):
        y_pred_probas = []
        for name, pipeline in self.pipelines.items():
            proba = pipeline.predict_proba(data)
            y_pred_probas.append(self.weights[name] * proba)
        return np.sum(y_pred_probas, axis=0) / sum(self.weights.values())

    def predict(self, data):
        self.logger.info("Predicted Processing")
        y_pred_proba = self.predict_proba(data)
        y_pred_lab = np.argmax(y_pred_proba, axis=1) + 1  # convert to label
        return y_pred_lab

    def save(self, path):
        self.logger.info(f"Save model to {path}")
        os.makedirs(path, exist_ok=True)
        for pipeline in self.pipelines.values():
            pipeline.save(path)

    @staticmethod
    def load(path, model_settings, logger=logging):
        pipe = EnsemblePipeline.__new__(EnsemblePipeline)
        pipe.pipelines = {
            model.name: Pipeline.load(path, model)
            for model in model_settings
        }
        pipe.weights = {
            model.name: model.weight
            for model in model_settings
        }
        pipe.logger = logger
        return pipe
