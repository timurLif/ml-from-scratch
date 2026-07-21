import numpy as np


class LinearRegression:
    def __init__(self, penalty=None, regulation_power=0.1, learning_rate=0.1):
        self.penalty = penalty
        self.regulation_power = regulation_power
        self.learning_rate = learning_rate
        self.w = None

    def calc_penalty_grad(self):
        if self.penalty == 'l2':
            grad = 2 * self.w
        elif self.penalty == 'l1':
            grad = np.sign(self.w)
        else:
            return np.zeros_like(self.w)
        
        # Не штрафуем интерцепт (w_0)
        grad[0] = 0
        return grad

    def calc_penalty_loss(self):
        if self.w is None:
            return 0.0
        
        w_without_intercept = self.w[1:]

        if self.penalty == 'l2':
            return np.sum(w_without_intercept ** 2)
        elif self.penalty == 'l1':
            return np.sum(np.abs(w_without_intercept))
        else:
            return 0.0

    def _compute_loss(self, X, y):
        l = X.shape[0]
        n_targets = y.shape[1] if y.ndim > 1 else 1
        
        error = X @ self.w - y
        mse_loss = np.trace(error.T @ error) / (l * n_targets) if y.ndim > 1 else (error.T @ error) / l
        reg_loss = (self.regulation_power * self.calc_penalty_loss()) / n_targets
        
        return mse_loss + reg_loss

    def fit(self, X, y, iteration_number=1000, reduction_factor=0.001, 
            tol_loss=10**(-5), tol_grad=10**(-4), tol_weights=10**(-6)):
        
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        y = np.array(y)
        
        l, n_features = X.shape
        
        if y.ndim > 1:
            n_targets = y.shape[1]
            self.w = np.zeros((n_features, n_targets))
        else:
            self.w = np.zeros(n_features)
        
        current_loss = self._compute_loss(X, y)

        for i in range(1, iteration_number + 1):
            current_learning_rate = self.learning_rate * np.exp(-reduction_factor * i)

            prev_w = np.copy(self.w)
            prev_loss = current_loss

            penalty_grad = self.calc_penalty_grad()

            grad = (2 / l) * X.T @ (X @ self.w - y) + self.regulation_power * penalty_grad
            self.w = self.w - current_learning_rate * grad

            current_loss = self._compute_loss(X, y)

            if (np.abs(current_loss - prev_loss) < tol_loss) \
                or (np.linalg.norm(grad) < tol_grad) \
                or (np.linalg.norm(self.w - prev_w) < tol_weights):
                break
    
    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w