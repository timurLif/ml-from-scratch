import numpy as np
from tree.dt_regressor import DecisionTreeRegressor


class DTGradientBoostingClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def _softmax(self, score):
        score = score - np.max(score, axis=1, keepdims=True)
        exp = np.exp(score)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        self.trees_list = []
        self.n_classes = y.shape[1]
        pred = np.zeros(y.shape)
        pred_proba = self._softmax(pred)

        for idx in range(self.n_estimators):
            tree_ensemble = []
            
            for class_idx in range(y.shape[1]):

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf
                )
                
                tree.fit(X, y[:, class_idx] - pred_proba[:, class_idx])
                pred[:, class_idx] += tree.predict(X) / np.sqrt(idx + 1)
                tree_ensemble.append(tree)
            
            pred_proba = self._softmax(pred)
            self.trees_list.append(tree_ensemble)

    def predict_proba(self, X):
        pred = np.zeros((X.shape[0], self.n_classes))

        for lr, tree_ensemble in enumerate(self.trees_list):
            for class_idx, tree in enumerate(tree_ensemble):
                pred[:, class_idx] += tree.predict(X) / np.sqrt(lr + 1)
        
        return self._softmax(pred)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)