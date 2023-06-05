import logging
from collections import OrderedDict

from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier

from pipeline import EnsemblePipeline
from utils import hidden_message
from constant import BRF, LGBM, TAB, BBC


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
