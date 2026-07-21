import numpy as np


class LogisticRegression:
    def __init__(self, penalty=None, learning_rate=0.1, regulation_power=0.1):
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.regulation_power = regulation_power
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
    
    def fit(self, X, y, iteration_number=1000, reduction_factor=0.001,
            tol_loss=10**(-5), tol_grad=10**(-4), tol_weights=10**(-6)):
        
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]
        y = np.array(y)


        l, n_features = X.shape
        if y.ndim > 1:
            n_targets = y.shape[1]
        else:
            n_targets = len(np.unique(y))
            y_one_hot = np.zeros((l, n_targets))
            y_one_hot[np.arange(l), y] = 1
            y = y_one_hot
        
        self.w = np.zeros((n_features, n_targets))


        logits = X @ self.w
        stabilized_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(stabilized_logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        loss = -1/l * np.sum(y * np.log(softmax + 1e-15)) + (self.regulation_power * self.calc_penalty_loss()) / n_targets


        for i in range(1, iteration_number + 1):
            current_learning_rate = self.learning_rate * np.e**(-reduction_factor * i)
            prev_w = np.copy(self.w)
            prev_loss = loss

            penalty_grad = self.calc_penalty_grad()
            grad = (X.T @ (softmax - y)) / l + self.regulation_power * penalty_grad
            self.w = self.w - current_learning_rate * grad

            logits = X @ self.w
            stabilized_logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(stabilized_logits)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            loss = -1/l * np.sum(y * np.log(softmax + 1e-15)) + (self.regulation_power * self.calc_penalty_loss()) / n_targets

            if (np.abs(loss - prev_loss) < tol_loss) \
                or (np.linalg.norm(grad) < tol_grad) \
                or (np.linalg.norm(self.w - prev_w) < tol_weights):
                break

    def predict_probabilities(self, X):
        X = np.array(X)
        X = np.c_[np.ones(X.shape[0]), X]

        logits = X @ self.w
        stabilized_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(stabilized_logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return softmax
    
    def predict(self, X):
        return np.argmax(self.predict_probabilities(X), axis=1)