import numpy as np


def mean_squared_error(real, pred):
    return np.mean((real - pred)**2)

def root_mean_squared_error(real, pred):
    return np.sqrt(mean_squared_error(real, pred))

def mean_absolute_error(real, pred):
    return np.mean(np.abs(real - pred))

def r2_score(real, pred):
    dividend = np.sum((real - pred)**2)
    divisor =  np.sum((real - np.mean(real)) ** 2)

    if divisor == 0:
        return 1.0 if np.all(real == pred) else 0.0
    
    return 1 - (dividend / divisor)