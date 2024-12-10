from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.preprocessing import FunctionTransformer


from save_results import remove_unserializable_keys, save_result
from filter_data import filter_dataset, get_scale
from all_factories import (
                            regressor_factory,
                            regressor_search_space,
                            transforms)



from process_scoring import (process_results)
from split import (
    cross_validate_regressor,
    get_incremental_split,
    get_feature_importance,
    get_generalizability_score
)

HERE: Path = Path(__file__).resolve().parent


def set_globals(Test: bool=False) -> None:
    global SEEDS, N_FOLDS, BO_ITER
    if not Test:
        SEEDS = [6, 13, 42, 69, 420, 1234567890, 473129]
        N_FOLDS = 5
        BO_ITER = 42
    else:
        SEEDS = [42,13]
        N_FOLDS = 2
        BO_ITER = 1



def train_regressor(
    dataset: pd.DataFrame,
    features_impute: Optional[list[str]],
    special_impute: Optional[str],
    structural_features: Optional[list[str]],
    numerical_feats: Optional[list[str]],
    unroll: Union[dict[str, str], list[dict[str, str]], None],
    regressor_type: str,
    target_features: str,
    transform_type: str,
    generalizability:bool,
    hyperparameter_optimization: bool=True,
    Test:bool=False,
    ) -> None:
        """
        you should change the name here for prepare
        """
            #seed scores and seed prediction
        set_globals(Test)
        scores, predictions, data_shape,feature_importance = _prepare_data(
                                                            dataset=dataset,
                                                            features_impute= features_impute,
                                                            special_impute= special_impute,
                                                            structural_features=structural_features,
                                                            unroll=unroll,
                                                            numerical_feats = numerical_feats,
                                                            target_features=target_features,
                                                            regressor_type=regressor_type,
                                                            transform_type=transform_type,
                                                            generalizability=generalizability,
                                                            hyperparameter_optimization=hyperparameter_optimization,
                                                            )
        scores = process_results(scores, data_shape, generalizability)
  
        return scores, predictions, data_shape
        



def _prepare_data(
    dataset: pd.DataFrame,
    target_features: str,
    regressor_type: str,
    generalizability:bool,
    structural_features: Optional[list[str]]=None,
    numerical_feats: Optional[list[str]]=None,
    unroll: Union[dict, list, None] = None,
    transform_type: str = "Standard",
    hyperparameter_optimization: bool = True,
    **kwargs,
    ) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:


    """
    here you should change the names
    """



    X, y, unrolled_feats, data_shape = filter_dataset(
                                        raw_dataset=dataset,
                                        structure_feats=structural_features,
                                        scalar_feats=numerical_feats,
                                        target_feats=target_features,
                                        dropna = True,
                                        unroll=unroll,
                                        )

    # Pipline workflow here and preprocessor
    # TODO: add scaler.
    # preprocessor: Pipeline = preprocessing_workflow(imputer=imputer,
    #                                                 feat_to_impute=features_impute,
    #                                                 representation = representation,
    #                                                 numerical_feat=numerical_feats,
    #                                                 structural_feat = unrolled_feats,
    #                                                 special_column=special_impute,
    #                                                 scaler=transform_type)
    


    preprocessor.set_output(transform="pandas")
    score, predication, feature_importance = run(
                                                X,
                                                y,
                                                preprocessor=preprocessor,
                                                regressor_type=regressor_type,
                                                transform_type=transform_type,
                                                hyperparameter_optimization=hyperparameter_optimization,
                                                **kwargs,
                                                )
    
    combined_prediction_ground_truth = pd.concat([predication, y.reset_index(drop=True)], axis=1)
    return score, combined_prediction_ground_truth, data_shape

def run(
        X,
        y,
        preprocessor: Union[ColumnTransformer, Pipeline],
        regressor_type: str,
        transform_type: str, 
        hyperparameter_optimization: bool = True,
        generalizability:bool=False,
        feat_importance:bool=False,
        ) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:

    seed_scores: dict[int, dict[str, float]] = {}
    seed_predictions: dict[int, np.ndarray] = {}
    # if 'Rh (IW avg log)' in target_features:
    #     y = np.log10(y)
    

    for seed in SEEDS:
      cv_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
      y_transform = get_target_transformer(transform_type)

      y_transform_regressor = TransformedTargetRegressor(
            regressor=regressor_factory[regressor_type],
            transformer=y_transform,
        )
      regressor :Pipeline= Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", y_transform_regressor),
                        ])

      # set_output on dataframe
      regressor.set_output(transform="pandas")
      #TODO: modify generalizability
      if hyperparameter_optimization:
            best_estimator, regressor_params = _optimize_hyperparams(
                X,
                y,
                cv_outer=cv_outer,
                seed=seed,
                regressor_type=regressor_type,
                regressor=regressor,
            )
            scores, predictions = cross_validate_regressor(
                best_estimator, X, y, cv_outer
            )
            scores["best_params"] = regressor_params


      else:
            scores, predictions = cross_validate_regressor(regressor, X, y, cv_outer)



      if generalizability:
          train_sizes, train_scores, test_scores = get_incremental_split(regressor,
                                                                                X,
                                                                                y,
                                                                                cv_outer,
                                                                                steps=0.2,
                                                                                random_state=seed)
          scores = get_generalizability_score(X,
                                                scores,
                                                train_sizes,
                                                train_scores,
                                                test_scores,
                                                )
          
      if feat_importance:    
            importance = get_feature_importance(X, scores, seed)
            feature_importances.append(importance)

      scores.pop('estimator', None)
      seed_scores[seed] = scores
      seed_predictions[seed] = predictions.flatten()
      # saving generalizability scores

    seed_predictions: pd.DataFrame = pd.DataFrame.from_dict(
                      seed_predictions, orient="columns")
    if feat_importance:
        seed_importance = pd.concat(feature_importances, axis=0, ignore_index=True)
    else:
        seed_importance =None
    return seed_scores, seed_predictions,seed_importance
            


# def _pd_to_np(data):
#     if isinstance(data, pd.DataFrame):
#         return data.values
#     elif isinstance(data, np.ndarray):
#         return data
#     else:
#         raise ValueError("Data must be either a pandas DataFrame or a numpy array.")


def _optimize_hyperparams(
    X, y, groups:np.array, seed: int, regressor_type: str, regressor: Pipeline) -> tuple:

    # Splitting for outer cross-validation loop
    estimators: list[BayesSearchCV] = []
    cv_inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train_index, _ in cv_inner.split(X, y, groups):

        X_train = split_for_training(X, train_index)
        y_train = split_for_training(y, train_index)
        # print(X_train)
        # Splitting for inner hyperparameter optimization loop
        print("\n\n")
        print(
            "OPTIMIZING HYPERPARAMETERS FOR REGRESSOR", regressor_type, "\tSEED:", seed
        )
        # Bayesian hyperparameter optimization
        bayes = BayesSearchCV(
            regressor,
            regressor_search_space[regressor_type],
            n_iter=BO_ITER,
            cv=cv_inner,
            n_jobs=-1,
            random_state=seed,
            refit=True,
            scoring="r2",
            return_train_score=True,
        )
        bayes.fit(X_train, y_train)

        print(f"\n\nBest parameters: {bayes.best_params_}\n\n")
        estimators.append(bayes)

    # Extract the best estimator from hyperparameter optimization
    best_idx: int = np.argmax([est.best_score_ for est in estimators])
    best_estimator: Pipeline = estimators[best_idx].best_estimator_
    try:
        regressor_params: dict = best_estimator.named_steps.regressor.get_params()
        regressor_params = remove_unserializable_keys(regressor_params)
    except:
        regressor_params = {"bad params": "couldn't get them"}

    return best_estimator, regressor_params


def split_for_training(
    data: Union[pd.DataFrame, np.ndarray,pd.Series], indices: np.ndarray
) -> Union[pd.DataFrame, np.ndarray, pd.Series]:
    if isinstance(data, pd.DataFrame):
        split_data = data.iloc[indices]
    elif isinstance(data, np.ndarray):
        split_data = data[indices]
    elif isinstance(data, pd.Series):
        split_data = data.iloc[indices]
    else:
        raise ValueError("Data must be either a pandas DataFrame, Series, or a numpy array.")
    return split_data



def get_target_transformer(transformer) -> Pipeline:

        return Pipeline(steps=[
            ("log transform", FunctionTransformer(np.log10, inverse_func=lambda x: 10**x,
                                                  check_inverse=True, validate=False)),  # log10(x)
            ("y scaler", transforms[transformer])  # StandardScaler to standardize the log-transformed target
            ])


    


