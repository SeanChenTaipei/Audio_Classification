from src.constant import (
    BRF, LGBM, TABPFN,
    BRF_BBC, LGBM_BBC, TABPFN_BBC,
)


BRF_PARAMS = {
    "n_estimators": 500,
    "class_weight": "balanced_subsample",
}

LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "class_weight": "balanced",
    "objective": "multiclass",
}

TABPFN_PARAMS = {
    "N_ensemble_configurations": 100,
}

MODEL_PARAMS = {
    BRF: BRF_PARAMS,
    LGBM: LGBM_PARAMS,
    TABPFN: TABPFN_PARAMS,
    BRF_BBC: BRF_PARAMS,
    LGBM_BBC: LGBM_PARAMS,
    TABPFN_BBC: TABPFN_PARAMS,
}
