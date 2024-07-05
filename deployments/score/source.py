import modelbit, sys
from typing import *
from sklearn.linear_model._base import LinearRegression
import pandas as pd
import numpy as np

lreg = modelbit.load_value("data/lreg.pkl") # LinearRegression()

# main function
def score(lead_features: pd.DataFrame) -> np.ndarray:
    return lreg.predict(lead_features)

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = score(...)
#   print(result)