BRF_PARAMS = {
    "n_estimators": 500,
    "class_weight": "balanced_subsample",
}

LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "class_weight": "balanced",
    "objective": "multiclass",
}

TAB_PARAMS = {"N_ensemble_configurations": 100}
