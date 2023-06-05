import numpy as np


class Dataset(object):
    def __init__(self, X, y, ids):
        self.X = X
        self.y = y
        self.ids = ids

    @staticmethod
    def merge(d1, d2):
        X = np.concatenate([d1.X, d2.X], axis=0)
        y = np.concatenate([d1.y, d2.y], axis=0) if d1.y is not None and d2.y is not None else None
        ids = np.concatenate([d1.ids, d2.ids], axis=0)
        return Dataset(X=X, y=y, ids=ids)
