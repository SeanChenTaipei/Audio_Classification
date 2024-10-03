import numpy as np
from sklearn.utils import check_array

from src.pdict import PDict


class Dataset(PDict):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            kwargs[key] = check_array(value, dtype=None, ensure_2d=False)
            kwargs[key].setflags(write=False)  # assure immutability within numpy array
            assert kwargs[key].shape[0] == list(kwargs.values())[0].shape[0]
        super().__init__(**kwargs)

    @staticmethod
    def merge(d1, d2):
        X = np.concatenate([d1.X, d2.X], axis=0)
        y = (
            np.concatenate([d1.y, d2.y], axis=0)
            if d1.y is not None and d2.y is not None
            else None
        )
        ids = np.concatenate([d1.ids, d2.ids], axis=0)
        return Dataset(X=X, y=y, ids=ids)
