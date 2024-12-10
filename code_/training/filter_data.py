from typing import Callable, Optional, Union, Dict, Tuple

import pandas as pd
import numpy as np
from unrolling_utils import unrolling_factory


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

    if unroll:
        if isinstance(unroll, dict):
            structure_features: pd.DataFrame = unrolling_factory[
                unroll["representation"]](dataset[structure_feats], **unroll)
        else:
            raise ValueError(f"Unroll must be a dict, not {type(unroll)}")
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
    split_groups = get_group(raw_dataset)
    return training_features, targets, new_struct_feats, split_groups, training_test_shape


def get_group(df)-> np.array:
    return df['common_name'].to_numpy()
