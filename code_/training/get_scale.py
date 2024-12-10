from typing import Callable, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from sklearn.compose import ColumnTransformer
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from all_factories import (imputer_factory, 
                           representation_scaling_factory,
                           transforms)



def preprocessing_workflow(
                           numerical_feat: Optional[list] = None,
                           structural_feat: Optional[list] = None,
                           scaler: str = "Standard",
                           ) -> Pipeline:
    transformers = []
    if numerical_feat:
        transformers.append(
            ("numerical_scaling", transforms[scaler], numerical_feat)
            )
    # elif representation_scaling_factory[representation]['callable']:
    elif structural_feat:
         transformers.append(
            ("structural_scaling", transforms[scaler], structural_feat)
            )
        
    scaling = ("scaling features",
              ColumnTransformer(transformers=[*transformers], remainder="passthrough", verbose_feature_names_out=False)
              )
    return Pipeline([scaling])