import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from typing import Callable, Optional, Union, Dict, Tuple
import numpy as np
import sys
sys.path.append("../cleaning")
from argparse import ArgumentParser
from save_results import save_results

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training"/ "training_dataset.pkl"
w_data = pd.read_pickle(training_df_dir)

TEST = True

def main_numerical_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    numerical_feats: Optional[list[str]],
) -> None:


    scores, predictions, feature_importance  = train_regressor(
                                            dataset=dataset,
                                            structural_features=None,
                                            unroll=None,
                                            numerical_feats=numerical_feats,
                                            target_features=["ofet.hole_mobility"],
                                            regressor_type=regressor_type,
                                            transform_type="Standard",
                                            hyperparameter_optimization=True,
                                            feat_importance=False,
                                            generalizability=True,
                                            Test=TEST,
                                            )
    
    save_results(scores,
                predictions=predictions,
                # importance_score= feature_importance,
                generalizability=True,
                representation= None,
                target_features=["ofet.hole_mobility"],
                regressor_type=regressor_type,
                numerical_feats=numerical_feats,
                transform_type="Standard",
                TEST=TEST,
                )


    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # imputer = "mean"
    # transform_type= "Standard"
    # target_features= ['Lp (nm)']
    
feat_list:list[str] = ["polymer size"
                           ]
models = "RF"
main_numerical_only(
                dataset=w_data,
                numerical_feats=feat_list,
                regressor_type=models

)



# def parse_arguments():
#     parser = ArgumentParser(description="Process some data for numerical-only regression.")
    
#     # Argument for regressor_type
#     parser.add_argument(
#         '--target_features',
#         choices=['Lp (nm)', 'Rg1 (nm)', 'Rh (IW avg log)'],  
#         required=True,
#         help="Specify a single target for the analysis."
#     )
    
#     parser.add_argument(
#         '--regressor_type', 
#         type=str, 
#         choices=['RF', 'DT', 'MLR', 'SVR', 'XGBR','KNN', 'GPR', 'NGB'], 
#         required=True, 
#         help="Regressor type required"
#     )

#     parser.add_argument(
#         '--numerical_feats',
#         type=str,
#         choices=['Mn (g/mol)', 'Mw (g/mol)', 'PDI', 'Temperature SANS/SLS/DLS/SEC (K)',
#                   'Concentration (mg/ml)','solvent dP',	'polymer dP',	'solvent dD',	'polymer dD',	'solvent dH',	'polymer dH', 'Ra',
#                   "abs(solvent dD - polymer dD)", "abs(solvent dP - polymer dP)", "abs(solvent dH - polymer dH)"],

#         nargs='+',  # Allows multiple choices
#         required=True,
#         help="Numerical features: choose"
#     )
    
#     parser.add_argument(
#         '--columns_to_impute',
#         type=str,
#         choices=['Mn (g/mol)', 'Mw (g/mol)', 'PDI', 'Temperature SANS/SLS/DLS/SEC (K)',
#                   'Concentration (mg/ml)','solvent dP',	'polymer dP',	'solvent dD',	'polymer dD',	'solvent dH',	'polymer dH', 'Ra'],

#         nargs='*',  # This allows 0 or more values
#         default=None,  
#         help="imputation features: choose"
#     )

#     parser.add_argument(
#         '--imputer',
#         choices=['mean', 'median', 'most_frequent',"distance KNN", None],  
#         nargs='?',  # This allows the argument to be optional
#         default=None,  
#         help="Specify the imputation strategy or leave it as None."
#     )

#     parser.add_argument(
#         '--special_impute',
#         choices=['Mw (g/mol)', None],  
#         nargs='?',  # This allows the argument to be optional
#         default=None,  # Set the default value to None
#         help="Specify the imputation strategy or leave it as None."
#     )

#     parser.add_argument(
#         "--transform_type", 
#         type=str, 
#         choices=["Standard", "Robust Scaler"], 
#         default= "Standard", 
#         help="transform type required"
#     )

#     parser.add_argument(
#         "--kernel", 
#         type=str,
#         default=None,
#         help='kernel for GP is optinal'
#     )
    

# if __name__ == "__main__":
#     args = parse_arguments()
    
#     print(args.regressor_type)
#     print(type(args.regressor_type))

#     main_numerical_only(
#         dataset=w_data,
#         regressor_type=args.regressor_type,
#         kernel=args.kernel,
#         target_features=[args.target_features],  # Already a list from `choices`, no need to wrap
#         transform_type=args.transform_type,
#         hyperparameter_optimization=True,
#         columns_to_impute=args.columns_to_impute,  # Already a list
#         special_impute=args.special_impute,
#         numerical_feats=args.numerical_feats,  # Already a list
#         imputer=args.imputer,
#         cutoff=None,  # Optional cutoff value
#     )

    # main_numerical_only(
    #     dataset=w_data,
    #     regressor_type="GPR",
    #     kernel= "matern",
    #     target_features=['Rh (IW avg log)'],  # Can adjust based on actual usage
    #     transform_type="Standard",
    #     hyperparameter_optimization=True,
    #     columns_to_impute=None,
    #     special_impute=None,
    #     numerical_feats=['polymer dH'],
    #     imputer=None,
    #     cutoff=None)

    # columns_to_impute: list[str] = ["PDI","Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]
    # special_column: str = "Mw (g/mol)"
    # numerical_feats: list[str] = ["Mn (g/mol)", "Mw (g/mol)", "PDI", "Temperature SANS/SLS/DLS/SEC (K)","Concentration (mg/ml)"]

# "intensity weighted average over log(Rh (nm))"



