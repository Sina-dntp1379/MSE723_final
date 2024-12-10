from sklearn.linear_model import ( ElasticNet)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from xgboost import XGBRegressor
from ngboost import NGBRegressor


from sklearn.preprocessing import (StandardScaler,
                                   RobustScaler)

from skopt.space import Integer, Real, Categorical
from typing import Callable, Optional, Union, Dict





transformers: dict[str, Callable] = {
    None:                None,
    "Standard":          StandardScaler(),
    "Robust Scaler":      RobustScaler(),
}


regressor_factory: dict[str, type]={
    "RF": RandomForestRegressor(),
    "ElasticNet": ElasticNet(), 
    "DT": DecisionTreeRegressor(),
    "XGBR": XGBRegressor(),
    "NGB": NGBRegressor(),
}


regressor_search_space = {

    "ElasticNet": {
        "regressor__regressor__alpha": Real(1e-4, 1e3, prior="log-uniform"),
        'regressor__regressor__l1_ratio': Real(0, 1) 
    },

    "RF": {
        "regressor__regressor__n_estimators": Integer(10, 2000, prior="log-uniform"),
        "regressor__regressor__min_samples_split": Real(0.05, 0.99),
        "regressor__regressor__min_samples_leaf": Real(0.05, 0.99),
        "regressor__regressor__max_features": Categorical(["sqrt", "log2"]),
    },

    "DT": {
        "regressor__regressor__min_samples_split": Real(0.05, 0.99),
        "regressor__regressor__min_samples_leaf": Real(0.05, 0.99),
        "regressor__regressor__max_features": Categorical([None,"sqrt", "log2"]),
        "regressor__regressor__ccp_alpha": Real(0.05, 0.99),
    },

    "NGB": {
        "regressor__regressor__n_estimators": Integer(50, 2000, prior="log-uniform"),
        "regressor__regressor__learning_rate": Real(1e-6, 1e-3, prior="log-uniform"),
        "regressor__regressor__natural_gradient": [True],
        "regressor__regressor__verbose": [False],
    },
}