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

TEST = False

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


# feat_list:list[str] = ["environmental parameters" "polymer size"
#                            ]
# models = "RF"
# main_numerical_only(
#                 dataset=w_data,
#                 numerical_feats=feat_list,
#                 regressor_type=models

# )


# passing to HPC Hazel

def parse_arguments():
    parser = ArgumentParser(description="Process some data for numerical-only regression.")
    

    
    parser.add_argument(
        '--regressor_type', 
        type=str, 
        choices=['RF', 'DT', 'ElasticNet', 'XGBR', 'NGB'], 
        required=True, 
        help="Regressor type required"
    )

    # parser.add_argument(
    #     '--numerical_feats',
    #     type=str,
    #     required=True,
    #     help="Numerical features: choose"
    # )
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()

    main_numerical_only(
        dataset=w_data,
        regressor_type=args.regressor_type,
        numerical_feats=["environmental parameters", "polymer size"],
    )




