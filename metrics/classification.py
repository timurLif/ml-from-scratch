import numpy as np


def _true_positive(y_true, y_pred):
    return np.sum([
        1 if (y_true[idx] == 1 and y_pred[idx] == 1) else 0
        for idx in range(len(y_true))
    ])

def _false_positive(y_true, y_pred):
    return np.sum([
        1 if (y_true[idx] == 0 and y_pred[idx] == 1) else 0
        for idx in range(len(y_true))
    ])

def _false_negative(y_true, y_pred):
    return np.sum([
        1 if (y_true[idx] == 1 and y_pred[idx] == 0) else 0
        for idx in range(len(y_true))
    ])


def precision_score(y_true, y_pred):
    tp = _true_positive(y_true, y_pred)
    fp = _false_positive(y_true, y_pred)

    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    tp = _true_positive(y_true, y_pred)
    fn = _false_negative(y_true, y_pred)

    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return 2 * precision * recall / (precision + recall)
