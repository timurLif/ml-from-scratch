import numpy as np
from ..tree.dt_regressor import DecisionTreeRegressor


class RandomDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, max_depth=None, min_samples_leaf=1):
        super().__init__(max_depth, min_samples_leaf)
    
    def _best_split(self, X, y):
        best_feature_idx = None
        best_threshold = None
        max_gain = 0.0

        max_features = max(1, int(np.sqrt(X.shape[1])))
        random_feature_list = np.random.choice(X.shape[1], size=max_features, replace=False)
    
        for feature_idx in random_feature_list:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask
                
                len_left, len_right = np.sum(left_mask), np.sum(right_mask)

                if len_left < self.min_samples_leaf or len_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                current_gain = super().variance_reduction(y, y_left, y_right)
                if current_gain > max_gain:
                    max_gain = current_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold


class RandomForestRegressor:
    def __init__(self, n_estimators, max_depth=None, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):

        n_samples = X.shape[0]
        self.trees_list = []

        for _ in range(self.n_estimators):
            idx_list = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[idx_list], y[idx_list]

            tree = RandomDecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            tree.fit(X_boot, y_boot)

            self.trees_list.append(tree)
            
    def predict(self, X):
        X = np.array(X, dtype=np.float64)

        predictions = [tree.predict(X) for tree in self.trees_list]
        return np.mean(np.array(predictions), axis=0)
