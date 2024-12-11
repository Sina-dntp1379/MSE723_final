import pandas as pd
from pathlib import Path
from training_utils import train_regressor
from all_factories import radius_to_bits
import sys
# import numpy as np
from typing import Callable, Optional, Union, Dict, Tuple
sys.path.append("../cleaning")
# from clean_dataset import open_json
from argparse import ArgumentParser
from save_results import save_results


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training"/ "training_dataset.pkl"
w_data = pd.read_pickle(training_df_dir)

TEST=False


def main_ECFP_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    radius: int,
    generalizability:bool,
    hyperparameter_optimization:bool,

) -> None:
    representation: str = "ECFP"
    n_bits = radius_to_bits[radius]
    structural_features: list[str] = [
        f"poly_smile_{representation}{2 * radius}_count_{n_bits}bits"
    ]
    unroll_single_feat = {
        "representation": representation,
        "radius": radius,
        "n_bits": n_bits,
        "col_names": structural_features,
    }
    scores, predictions,  feature_imp = train_regressor(
                                        dataset=dataset,
                                        structural_features=structural_features,
                                        unroll=unroll_single_feat,
                                        numerical_feats=None,
                                        target_features=target_features,
                                        regressor_type=regressor_type,
                                        transform_type=transform_type,
                                        hyperparameter_optimization=hyperparameter_optimization,
                                        feat_importance=None,
                                        generalizability=generalizability,
                                        Test=TEST
                                        )
    save_results(scores,
            predictions=predictions,
            # importance_score=feature_imp,
            generalizability=generalizability,
            representation= representation,
            # radius= radius,
            target_features=target_features,
            regressor_type=regressor_type,
            TEST=TEST,
            transform_type=transform_type
            )


def main_Mordred_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    generalizability:bool,
    hyperparameter_optimization:bool,

) -> None:
    representation: str = "Mordred"
    structural_features: list[str] = [f"poly_smile_{representation}"]
    unroll_single_feat = {"representation": representation,
                          "col_names": structural_features}

    scores, predictions, feature_imp  = train_regressor(
                                    dataset=dataset,
                                    structural_features=structural_features,
                                    unroll=unroll_single_feat,
                                    numerical_feats=None,
                                    target_features=target_features,
                                    regressor_type=regressor_type,
                                    transform_type=transform_type,
                                    hyperparameter_optimization=hyperparameter_optimization,
                                    feat_importance=None,
                                    generalizability=generalizability,
                                    Test=TEST
                                    )

    save_results(scores=scores,
                predictions=predictions,
                # importance_score=feature_imp,
                generalizability=generalizability,
                representation= representation,
                target_features=target_features,
                regressor_type=regressor_type,
                TEST=TEST,
                transform_type=transform_type
                )





def perform_model_ecfp(regressor_type:str,
                        ):
                
                main_ECFP_only(
                                dataset=w_data,
                                regressor_type= regressor_type,
                                target_features= ["ofet.hole_mobility"],
                                hyperparameter_optimization= True,
                                radius = 6,
                                generalizability=True,
                                transform_type="Standard"
                                )


def perform_model_mordred(regressor_type:str):
 
                main_Mordred_only(
                                dataset=w_data,
                                regressor_type= regressor_type,
                                target_features= ["ofet.hole_mobility"],
                                hyperparameter_optimization= True,
                                generalizability=True,
                                transform_type="Standard"
                                )




# perform_model_mordred('RF')
# perform_model_ecfp('RF')

def main():
    parser = ArgumentParser(description='Run models with specific parameters')

    # Subparsers for different models
    subparsers = parser.add_subparsers(dest='model', required=True, help='Choose a model to run')

    # Parser for ECFP model
    parser_ecfp = subparsers.add_parser('ecfp', help='Run the ECFP model')
    parser_ecfp.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')


    # Parser for Mordred model
    parser_mordred = subparsers.add_parser('mordred', help='Run the Mordred numerical model')
    parser_mordred.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')

    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate model based on the parsed arguments
    if args.model == 'ecfp':
        perform_model_ecfp(args.regressor_type)

    elif args.model == 'mordred':
        perform_model_mordred(args.regressor_type)

if __name__ == '__main__':
    main()

