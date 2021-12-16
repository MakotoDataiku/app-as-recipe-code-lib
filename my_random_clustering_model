from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class MyRandomClusteringModel(BaseEstimator):
    """This model assigns clusters randomly"""

    def fit(self, X):
        pass

    def predict(self, X):
        return pd.Series(np.random.choice([0, 1, 2], size=X.shape[0]))

    def fit_predict(self, X):
        return pd.Series(np.random.choice([0, 1, 2], size=X.shape[0]))
