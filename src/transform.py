import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.constant import TABULAR, WAVLM_ENCODE, WAV2VEC_ENCODE
from src.utils import read_json


class BaseTransform:
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
        data = read_json(data).values()
        return np.array(list(data))

    def transform(self, data):
        data = read_json(data).values()
        return np.array(list(data))


class FillNa(BaseTransform):
    def __init__(self):
        super(FillNa, self).__init__()

    def fit_transform(self, data):
        return data.fillna(0)

    def transform(self, data):
        return data.fillna(0)


TRANS_v1 = [
    (TABULAR, LoadTabularData()),
    (WAVLM_ENCODE, LoadAudioData()),
    (TABULAR, FillNa()),
    (WAVLM_ENCODE, PCA(n_components=9)),
]

TRANS_v2 = [
    (TABULAR, LoadTabularData()),
    (WAVLM_ENCODE, LoadAudioData()),
    (WAV2VEC_ENCODE, LoadAudioData()),
    (TABULAR, FillNa()),
    (WAVLM_ENCODE, PCA(n_components=9)),
    (WAV2VEC_ENCODE, PCA(n_components=1)),
]

TRANS = {
    "v1": TRANS_v1,
    "v2": TRANS_v2,
}
