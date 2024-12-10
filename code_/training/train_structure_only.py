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
from save_results import save_result


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"
RESULTS = Path = HERE.parent.parent / "results"

training_df_dir: Path = DATASETS/ "training"/ "training_dataset.pkl"
w_data = pd.read_pickle(training_df_dir)

TEST=False


# def train_regressor(
#     dataset: pd.DataFrame,
#     structural_features: Optional[list[str]],
#     numerical_feats: Optional[list[str]],
#     unroll: Union[dict[str, str], list[dict[str, str]], None],
#     regressor_type: str,
#     target_features: str,
#     transform_type: str,
#     generalizability:bool,
#     feat_importance:bool,
#     hyperparameter_optimization: bool=True,
#     Test:bool=False,
#     ) -> None:

def main_ECFP_only(
    dataset: pd.DataFrame,
    regressor_type: str,
    target_features: list[str],
    transform_type: str,
    hyperparameter_optimization: bool,
    radius: int,
    generalizability:bool,
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
    scores, predictions, data_shapes = train_regressor(
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
    save_result(scores,
            predictions=predictions,
            df_shapes=data_shapes,
            generalizability_score=None,
            representation= representation,
            radius= radius,
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
    hyperparameter_optimization: bool,
    generalizability:bool,

) -> None:
    representation: str = "Mordred"
    structural_features: list[str] = [f"poly_smile_{representation}"]
    unroll_single_feat = {"representation": representation,
                          "col_names": structural_features}

    scores, predictions, data_shapes  = train_regressor(
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

    save_result(scores=scores,
                predictions=predictions,
                df_shapes=data_shapes,
                generalizability_score=None,
                representation= representation,
                target_features=target_features,
                regressor_type=regressor_type,
                TEST=TEST,
                transform_type=transform_type
                )






# for radius in radii:
# for vector in vectors:
# radii = [3, 4, 5, 6]
# vectors = ['count', 'binary']
def perform_model_ecfp(regressor_type:str,
                        ):
                
                main_ECFP_only(
                                dataset=w_data,
                                regressor_type= regressor_type,
                                target_features= ["ofet.hole_mobility"],
                                hyperparameter_optimization= True,
                                radius = 6,
                                )


def perform_model_mordred(regressor_type:str):
 
                main_Mordred_only(
                                dataset=w_data,
                                regressor_type= regressor_type,
                                target_features= ["ofet.hole_mobility"],
                                hyperparameter_optimization= True,
                                )




# perform_model_ecfp('GPR',6,"count",'Rg1 (nm)', 'Monomer',kernel='tanimoto')
# perform_model_maccs()
# perform_model_mordred('RF')

# def main():
#     parser = ArgumentParser(description='Run models with specific parameters')

#     # Subparsers for different models
#     subparsers = parser.add_subparsers(dest='model', required=True, help='Choose a model to run')

#     # Parser for ECFP model
#     parser_ecfp = subparsers.add_parser('ecfp', help='Run the ECFP model')
#     parser_ecfp.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
#     parser_ecfp.add_argument('--radius', type=int, choices=[3, 4, 5, 6], default=3, help='Radius for ECFP (default: 3)')
#     parser_ecfp.add_argument('--vector', choices=['count', 'binary'], default='count', help='Type of vector (default: count)')
#     parser_ecfp.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
#     parser_ecfp.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
#                                                           'RRU Dimer', 'RRU Trimer'],
#                                                             default='Monomer', help='polymer unite representation')
#     parser_ecfp.add_argument('--kernel', default=None, help='kernel for GP is optinal')
#     parser_ecfp.add_argument('--transform_type', default='Standard', help='scaler required')


#     # Parser for MACCS model
#     parser_maccs = subparsers.add_parser('maccs', help='Run the MACCS numerical model')
#     parser_maccs.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
#     parser_maccs.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
#     parser_maccs.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
#                                                           'RRU Dimer', 'RRU Trimer'],
#                                                             default='Monomer', help='polymer unite representation')
#     parser_maccs.add_argument('--kernel', default=None, help='kernel for GP is optinal')
#     parser_maccs.add_argument('--transform_type', default='Standard', help='scaler required')

#     # Parser for Mordred model
#     parser_mordred = subparsers.add_parser('mordred', help='Run the Mordred numerical model')
#     parser_mordred.add_argument('--regressor_type', default='RF', help='Type of regressor (default: RF)')
#     parser_mordred.add_argument('--target', default='Rg1 (nm)', help='Target variable (default: Rg1 (nm))')
#     parser_mordred.add_argument('--oligo_type', choices=['Monomer', 'Dimer', 'Trimer', 'RRU Monomer',
#                                                           'RRU Dimer', 'RRU Trimer'],
#                                                             default='Monomer', help='polymer unite representation')
#     parser_mordred.add_argument('--kernel', default=None, help='kernel for GP is optinal')
#     parser_mordred.add_argument('--transform_type', default='Standard', help='scaler required')

#     # Parse arguments
#     args = parser.parse_args()

#     # Run the appropriate model based on the parsed arguments
#     if args.model == 'ecfp':
#         perform_model_ecfp(args.regressor_type, args.radius, args.vector, args.target, args.oligo_type, args.kernel, args.transform_type)
#     elif args.model == 'maccs':
#         perform_model_maccs(args.regressor_type, args.target, args.oligo_type, args.kernel, args.transform_type)
#     elif args.model == 'mordred':
#         perform_model_mordred(args.regressor_type, args.target, args.oligo_type, args.kernel, args.transform_type)

# if __name__ == '__main__':
#     main()

