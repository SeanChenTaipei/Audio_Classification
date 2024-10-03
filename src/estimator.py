import pickle
import os

from imblearn.ensemble import (
    BalancedBaggingClassifier,
    BalancedRandomForestClassifier,
)
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier

from src.constant import BRF, LGBM, TABPFN, BBC
from src.utils import hidden_message


class Estimator:
    def __init__(self, algo=None, params=None):
        self.algo = algo
        self.params = params
        self.model = None
        self.initialize()

    def initialize(self):
        if self.model is not None:
            return

        if self.algo is None:
            raise ValueError("Algorithm must be specified to initialize the model.")

        if BRF in self.algo:
            self.model = BalancedRandomForestClassifier(**self.params)

        elif LGBM in self.algo:
            self.model = LGBMClassifier(**self.params)

        elif TABPFN in self.algo:
            with hidden_message:
                self.model = TabPFNClassifier(**self.params)

        if BBC in self.algo:
            self.model = BalancedBaggingClassifier(estimator=self.model)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def save(self, path):
        with open(os.path.join(path, f"{self.algo}.pkl"), "wb") as f:
            pickle.dump(self.model, f)
            f.close()

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
            f.close()
        return model
