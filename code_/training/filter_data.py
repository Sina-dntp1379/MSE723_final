from typing import Callable, Optional, Union, Dict, Tuple
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from all_factories import transformers
import pandas as pd




def sanitize_dataset(training_feats, traget_faet):
    """
    Sanitize the training features and targets in case the target features contain NaN values.

    Args:
        training_features: Training features.
        targets: Targets.
        dropna: Whether to drop NaN values.
        **kwargs: Keyword arguments to pass to filter_dataset.

    Returns:
        Sanitized training features and targets.
    """
    traget_faet: pd.DataFrame = traget_faet.dropna()
    training_feats: pd.DataFrame =training_feats.loc[traget_faet.index]
    return training_feats, traget_faet



# def get_data(raw_dataset:pd.DataFrame,
#               feats:list,
#               target:str):

#     training_features:pd.DataFrame = raw_dataset[feats]
#     target:pd.DataFrame = raw_dataset[target]
#     training_features, target = sanitize_dataset(training_features,target)
#     training_test_shape: dict ={
#                                 "targets_shape": target.shape,
#                                 "training_features_shape": training_features.shape
#                                 }

#     return training_features, target, training_test_shape


def get_scale(feats:list,
               scaler_type:str)-> Pipeline:
    transformer = [("structural_scaling", transformers[scaler_type], feats)]
    scaling = [("scaling features",
              ColumnTransformer(transformers=[*transformer], remainder="passthrough", verbose_feature_names_out=False)
              )]
    
    return Pipeline(scaling)



def sanitize_dataset(
    training_features: pd.DataFrame, targets: pd.DataFrame, dropna: bool, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanitize the training features and targets in case the target features contain NaN values.

    Args:
        training_features: Training features.
        targets: Targets.
        dropna: Whether to drop NaN values.
        **kwargs: Keyword arguments to pass to filter_dataset.

    Returns:
        Sanitized training features and targets.
    """
    if dropna:
        targets: pd.DataFrame = targets.dropna()
        training_features: pd.DataFrame =training_features.loc[targets.index]
        return training_features, targets
    else:
        return training_features, targets


def filter_dataset(
    raw_dataset: pd.DataFrame,
    structure_feats: Optional[list[str]], # can be None
    scalar_feats: Optional[list[str]], # like conc, temp
    target_feats: list[str], # lp
    cutoff: Dict[str, Tuple[Optional[float], Optional[float]]],
    dropna: bool = True,
    unroll: Union[dict, list, None] = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Filter the dataset.

    Args:
        raw_dataset: Raw dataset.
        structure_feats: Structure features.
        scalar_feats: Scalar features.
        target_feats: Target features.

    Returns:
        Input features and targets.
    """
    # Add multiple lists together as long as they are not NoneType
    all_feats: list[str] = [
        feat
        for feat_list in [structure_feats, scalar_feats, target_feats]
        if feat_list
        for feat in feat_list
    ]

    dataset: pd.DataFrame = raw_dataset[all_feats]
    if cutoff:
        dataset = apply_cutoff(dataset,cutoff)

    if unroll:
        if isinstance(unroll, dict):
            structure_features: pd.DataFrame = unrolling_factory[
                unroll["representation"]](dataset[structure_feats], **unroll)
        elif isinstance(unroll, list):
            multiple_unrolled_structure_feats: list[pd.DataFrame] = []
            for unroll_dict in unroll:
                single_structure_feat: pd.DataFrame = filter_dataset(
                    dataset,
                    # structure_feats=unroll_dict["columns"],
                    structure_feats=unroll_dict["col_names"],
                    scalar_feats=[],
                    target_feats=[],
                    # dropna=dropna,
                    dropna=False,
                    unroll=unroll_dict,
                )[0]
                multiple_unrolled_structure_feats.append(single_structure_feat)
            structure_features: pd.DataFrame = pd.concat(
                multiple_unrolled_structure_feats, axis=1
            )
        else:
            raise ValueError(f"Unroll must be a dict or list, not {type(unroll)}")
    elif structure_feats:
        structure_features: pd.DataFrame = dataset[structure_feats]
    else:
        structure_features: pd.DataFrame = dataset[[]]

    if scalar_feats:
      scalar_features: pd.DataFrame = dataset[scalar_feats]
    else:
      scalar_features: pd.DataFrame = dataset[[]]

    training_features: pd.DataFrame = pd.concat(
        [structure_features, scalar_features], axis=1
    )

    targets: pd.DataFrame = dataset[target_feats]

    training_features, targets = sanitize_dataset(
        training_features, targets, dropna=dropna, **kwargs
    )

    # if not (scalars_available and struct_available):
    new_struct_feats: list[str] = structure_features.columns.tolist()
    training_test_shape: Dict ={
                                "targets_shape": targets.shape,
                                "training_features_shape": training_features.shape
                                }
    return training_features, targets, new_struct_feats, training_test_shape