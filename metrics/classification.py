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

def roc_auc_score(y_true, y_score):
    sort_idxs = np.argsort(y_score)[::-1]
    y_score = y_score[sort_idxs]
    y_true = y_true[sort_idxs]

    threshold_idxs = np.r_(np.where(np.diff(y_score) != 0)[0])

    tps = np.r_[0, np.cumsum(y_true)[threshold_idxs]]
    fps = np.r_[0, 1 + threshold_idxs - tps]

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    return np.trapezoid(tpr, fpr)