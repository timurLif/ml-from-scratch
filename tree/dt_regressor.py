import numpy as np


class TreeNode:
    def __init__(self, left=None, right=None, threshold=None, feature_idx=None, value=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature_idx = feature_idx
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def mse(self, y):
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)

    def variance_reduction(self, y, y_left, y_right):
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        return self.mse(y) - (weight_left * self.mse(y_left) + weight_right * self.mse(y_right))

    def _best_split(self, X, y):
        best_feature_idx = None
        best_threshold = None
        max_gain = -1.0

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask

                len_left, len_right = np.sum(left_mask), np.sum(right_mask)

                if len_left < self.min_samples_leaf or len_right < self.min_samples_leaf:
                    continue

                gain = self.variance_reduction(y, y[left_mask], y[right_mask])

                if gain > max_gain:
                    max_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, current_depth=0):
        n_samples = len(y)


        if ((self.max_depth is not None) and (current_depth >= self.max_depth)) or \
            (self.mse(y) < 1e-7) or \
            (n_samples < (self.min_samples_leaf * 2)):

            return TreeNode(value=np.mean(y))

        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            return TreeNode(value=np.mean(y))

        left_mask = X[:, feature_idx] < threshold
        right_mask = ~left_mask
        left_node = self._build_tree(X[left_mask], y[left_mask], current_depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], current_depth + 1)

        return TreeNode(
            left=left_node,
            right=right_node,
            threshold=threshold,
            feature_idx=feature_idx
        )

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.root = self._build_tree(X, y, current_depth=0)

    def _predict_row(self, node, x):
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] < node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return np.array([self._predict_row(self.root, x) for x in X])