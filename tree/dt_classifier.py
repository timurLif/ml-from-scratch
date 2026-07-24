import numpy as np


class TreeNode:
    def __init__(self, left=None, right=None, feature_idx=None, threshold=None, value=None):
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value

    @property
    def _is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def _build_tree(self, X, y, current_depth):

        if (current_depth >= self.max_depth) or \
            (len(y) < self.min_samples_leaf * 2) or \
            (len(np.unique(y)) == 1):
                
                return TreeNode(value=np.bincount(y).argmax())
                
        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            return TreeNode(value=np.bincount(y).argmax())

        left_mask = X[:, feature_idx] < threshold
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], current_depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], current_depth + 1)

        return TreeNode(left_node, right_node, feature_idx, threshold)

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

                y_left = y[left_mask]
                y_right = y[right_mask]

                if (len(y_left) < self.min_samples_leaf) or (len(y_right) < self.min_samples_leaf):
                    continue
                
                gain = self._information_gain(y, y_left, y_right)
                if gain > max_gain:
                    max_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold

    def _entropy(self, data):
        if len(data) == 0:
            return 0.0
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, data, data_left, data_right):
        w_left = len(data_left) / len(data)
        w_right = len(data_right) / len(data)
        return self._entropy(data) - (w_left * self._entropy(data_left) + w_right * self._entropy(data_right))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.root = self._build_tree(X, y, current_depth=0)

    def _predict_row(self, node, x):
        if node._is_leaf:
            return node.value
        
        if x[node.feature_idx] < node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        X = np.array(X)
        node = self.root
        
        return np.array([self._predict_row(node, x) for x in X])