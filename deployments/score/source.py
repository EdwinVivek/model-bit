import modelbit, sys
from typing import *
from sklearn.pipeline import Pipeline
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.ensemble._forest import RandomForestRegressor
import pandas as pd
import numpy as np

pipeline = modelbit.load_value("data/pipeline.pkl") # Pipeline(steps=[('col_transformer', ColumnTransformer(n_jobs=-1, transformers=[('cat_trans', Pipeline(steps=[('oneH', OneHotEncoder(sparse_output=False))]), ['season', 'yr', 'mnth', 'hr', 'holiday', '...

# main function
def score(lead_features: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(lead_features)

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = score(...)
#   print(result)