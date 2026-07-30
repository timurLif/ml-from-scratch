import numpy as np
from tree.dt_regressor import DecisionTreeRegressor


class DTGradientBoostingRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    # ----
    # slow (O(N**2))
    # def fit(self, X, y):
    #     self.trees_list = []

    #     for _ in range(self.n_estimators):

    #         tree = DecisionTreeRegressor(
    #             max_depth=self.max_depth,
    #             min_samples_leaf=self.min_samples_leaf
    #         )

    #         tree.fit(X, y - self.predict(X))
    #         self.trees_list.append(tree)
            
    # def predict(self, X):
    #     pred = np.zeros((self.n_estimators, X.shape[0]))

    #     for idx, tree in enumerate(self.trees_list):
    #         pred[idx] = (1 / (np.sqrt(idx + 1))) * tree.predict(X)
    
    #     return np.sum(pred, axis=0)
    # ----

    def fit(self, X, y):
        self.trees_list = []
        general_pred = np.zeros(len(y))

        for idx in range(self.n_estimators):

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
            
            tree.fit(X, y - general_pred)
            general_pred += (1 / np.sqrt(idx + 1)) * tree.predict(X)

            self.trees_list.append(tree)
    

    def predict(self, X):
        pred = np.zeros(X.shape[0])

        for idx, tree in range(self.trees_list):
            pred += (1 / np.sqrt(idx + 1)) * tree.predict(X)

        return pred