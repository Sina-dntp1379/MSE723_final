import json
from pathlib import Path
from types import NoneType
from typing import Dict, Optional,Tuple

import numpy as np
import pandas as pd

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent.parent


feature_abbrev: Dict[str, str] = {
    "ofet.hole_mobility":          "hole mobility",

}

def remove_unserializable_keys(d: Dict) -> Dict:
    """Remove unserializable keys from a dictionary."""
    # for k, v in d.items():
    #     if not isinstance(v, (str, int, float, bool, NoneType, tuple, list, np.ndarray, np.floating, np.integer)):
    #         d.pop(k)
    #     elif isinstance(v, dict):
    #         d[k] = remove_unserializable_keys(v)
    # return d
    new_d: dict = {k: v for k, v in d.items() if
                   isinstance(v, (str, int, float, bool, NoneType, np.floating, np.integer))}
    return new_d


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return super(NumpyArrayEncoder, self).default(obj)
        


def _save(scores: Optional[Dict[int, Dict[str, float]]],
          predictions: Optional[pd.DataFrame],
          results_dir: Path,
          regressor_type: str,
          representation: str,
          numerical_feats: Optional[list[str]],
          transform_type:Optional[str]=None,
          generalizability:bool=True,
          ) -> None:
    
    results_dir.mkdir(parents=True, exist_ok=True)
    fname_root = f"{regressor_type}_{transform_type}"

    num_feats = "-".join(feature_abbrev.get(key,key) for key in numerical_feats) if numerical_feats else None
    fname_root = f"({num_feats})_{fname_root}" if num_feats else fname_root
    fname_root = f"({representation})_{fname_root}" if representation else fname_root
    
    # fname_root = f"{fname_root}_{regressor_type}_{transform_type}"
    fname_root = f"{fname_root}_generalizability" if generalizability else fname_root

    print("Filename:", fname_root)
    
    if scores:
        scores_file: Path = results_dir / f"{fname_root}_scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)
        print(scores_file)

    if predictions is not None and not predictions.empty:
        predictions_file: Path = results_dir / f"{fname_root}_predictions.csv"
        predictions.to_csv(predictions_file, index=False)
        print(predictions_file)
    
    print('Done Saving scores!')


def save_results(                
                # importance_score,
                scores:Optional[Dict[int, Dict[str, float]]]=None,
                predictions: Optional[pd.DataFrame]=None,
                target_features: list=None,
                regressor_type: str=None,
                generalizability:bool=True,
                TEST : bool =True,
                representation: str=None,
                numerical_feats: Optional[list[str]]=None,
                output_dir_name: str = "results",
                transform_type:Optional[str]=None,

                 ) -> None:
    
    targets_dir: str = "-".join([feature_abbrev.get(target, target) for target in target_features])
    feature_ids = []
        

    if numerical_feats:
        feature_ids.append('scaler')
    elif representation:
        feature_ids.append('structural')
    features_dir: str = "_".join(feature_ids)
    print(features_dir)

    
    f_root_dir = f"target_{targets_dir}"

    results_dir: Path = ROOT / output_dir_name / f_root_dir
    results_dir: Path = results_dir / "test" if TEST else results_dir
    results_dir: Path = results_dir / features_dir


    _save(scores=scores,
          predictions=predictions,
          results_dir=results_dir,
          regressor_type=regressor_type,
          representation=representation,
          numerical_feats=numerical_feats,
          transform_type=transform_type,
          generalizability=generalizability,
          )