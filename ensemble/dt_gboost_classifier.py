import numpy as np
from tree.dt_regressor import DecisionTreeRegressor


class DTGradientBoostingClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    