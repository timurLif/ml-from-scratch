import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        self.std_ = np.std(X, axis=0)
        self.mean_ = np.mean(X, axis=0)

    def transform(self, X):
        return (X - self.mean_) / self.std_
     
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_
    

class MinMaxScaler():
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
    
    def transform(self, X):
        return (X - self.min_) / (self.max_ - self.min_)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X * (self.max_ - self.min_) + self.min_