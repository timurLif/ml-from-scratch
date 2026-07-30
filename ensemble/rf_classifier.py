import numpy as np
from tree.dt_classifier import DecisionTreeClassifier


class RandomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth, min_samples_leaf):
        super().__init__(max_depth, min_samples_leaf)
    
    def _best_split(self, X, y):
        best_feature_idx = None
        best_threshold = None
        max_gain = -1.0

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

                current_gain = super()._information_gain(y, y_left, y_right)
                if current_gain > max_gain:
                    max_gain = current_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold


class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        self.trees_list = []

        for _ in range(self.n_estimators):
            samples_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[samples_idx], y[samples_idx]
            
            tree = RandomDecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_boot, y_boot)
            self.trees_list.append(tree)

    def predict(self, X):
        X = np.array(X)
        
        predicts = np.array([tree.predict(X) for tree in self.trees_list])

        result = []
        for i in range(X.shape[0]):
            votes = predicts[:, i]
            most_common = np.bincount(votes).argmax()
            result.append(most_common)
        
        return np.array(result)
