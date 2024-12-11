from typing import Callable, Optional, Union
import pandas as pd




def unroll_lists_to_columns(df: pd.DataFrame, unroll_cols: list[str], new_col_names: list[str]) -> pd.DataFrame:
    """
    Unroll a list of lists into columns of a DataFrame.

    Args:
        df: DataFrame to unroll.
        unroll_cols: List of columns containing list to unroll.
        new_col_names: List of new column names.

    Returns:
        DataFrame with unrolled columns.
    """
    rolled_cols: pd.DataFrame = df[unroll_cols]
    # rolled_cols: pd.DataFrame = df
    unrolled_df: pd.DataFrame = pd.concat([rolled_cols[col].apply(pd.Series) for col in rolled_cols.columns], axis=1)
    unrolled_df.columns = new_col_names
    return unrolled_df


def unroll_ECFP(df: pd.DataFrame, col_names: list[str], 
                radius: int = 6, n_bits: int = 0,**kwargs) -> pd.DataFrame:
    
    new_ecfp_col_names: list[str] = [f"ECFP{2 * radius}_count_bit{i}" for i in range(n_bits)]
    new_df: pd.DataFrame = unroll_lists_to_columns(df, col_names, new_ecfp_col_names)
    return new_df   


def unroll_Mordred_descriptors(df: pd.DataFrame, col_names: list[str],
                        **kwargs) -> pd.DataFrame:
    
    descriptors: pd.Series = df[col_names].squeeze()
    mordred_descriptors_urolled: pd.DataFrame = pd.DataFrame.from_records(descriptors)
    mordred_descriptors: pd.DataFrame = mordred_descriptors_urolled.rename(columns=lambda x: f"Monomer Mordred {x}")
    return mordred_descriptors


radius_to_bits: dict[int, int] = {3: 512, 4: 1024, 5: 2048, 6: 4096}

unrolling_factory: dict[str, Callable] = {
                                          "ECFP":                 unroll_ECFP,
                                          "Mordred":              unroll_Mordred_descriptors,
                                          }




unrolling_feature_factory: dict[str, list[str]] = {
                                                "polymer size":     ['mn', 'dispersity'],
                                                "single solvent descriptors":  ['concentration', 'solvent dP',
                                                                        'polymer dP', 'solvent dD', 'polymer dD',
                                                                        'solvent dH', 'polymer dH',
                                                                        ],
                                                "hsp descriptors": ["Ra"],
                                                "pair solvent descriptors": ['abs(solvent dD - polymer dD)',
                                                                              'abs(solvent dP - polymer dP)',
                                                                                'abs(solvent dH - polymer dH)'],

                                                "device parameters":    ['dielectric_thickness','channel_length', 'channel_width',
                                                                          'deposition_type encoded', 'electrode_configuration encoded','postprocess.annealing.temperature'],
                                                
                                                "environmental parameters":     ['params.environment encoded', 'ofet.environment encoded'],
                                                "selected features": ['solvent dP',
                                                                        'polymer dP', 'solvent dD', 'polymer dD',
                                                                        'solvent dH', 'polymer dH',
                                                                        'params.environment encoded', 'ofet.environment encoded',
                                                                        'deposition_type encoded','mn', 'dispersity']
                                                 }

def unroll_features(rolled_features:list[str])-> list:
    if rolled_features==None:
        return None
    else:
        unrolled_features =   [feats for features in rolled_features for feats in unrolling_feature_factory[features]]
        return unrolled_features





