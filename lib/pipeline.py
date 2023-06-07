import numpy as np
import pandas as pd

from copy import deepcopy
from dataset import Dataset
from constant import TABULAR, LABEL_COLUMN


class DataPipeline():
    def __init__(self, transform_list):
        self.data_type = sorted(list(set([t[0] for t in transform_list])), key=len)
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
        for name, func in self.transform_list:
            data[name] = func.fit_transform(data[name])
        X, y = self.get_X_and_y(data)

        X = np.array(X)
        y = np.array(y)
        ids = np.array(data[TABULAR].index)

        return Dataset(X=X, y=y, ids=ids)

    def transform(self, data):
        data = deepcopy(data)
        for name, func in self.transform_list:
            data[name] = func.fit_transform(data[name])
        X = self.concat(data)
        return Dataset(X=X, y=None, ids=data[TABULAR].index)


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
