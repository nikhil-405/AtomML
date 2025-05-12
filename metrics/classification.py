import numpy as np
from core.tensor import Tensor
from typing import Union

Arr = Union[Tensor, np.ndarray]

# handles the data stored in core.Tensor class
def _to_numpy(x):
    if isinstance(x, Tensor):
        return x.data
    return np.array(x)

# Accuracy
def accuracy_score(y_true: Arr, y_pred: Arr) -> float:
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    return np.mean(y_true == y_pred)

def precision_score(y_true: Arr, y_pred: Arr, average : str = 'binary') -> float:
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    def _tp_fp_fn(label):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        return tp, fp, fn

    precisions = []
    for lbl in labels:
        tp, fp, _ = _tp_fp_fn(lbl)
        if (tp + fp):
           precision = tp / (tp + fp)
        else:
            precision = 0.0
        precisions.append(precision)

    if average == 'binary':
        return precisions[-1]
    elif average == 'macro':
        return np.mean(precisions)
    else:
        raise ValueError("average must be 'binary' or 'macro'")

def recall_score(y_true: Arr, y_pred: Arr, average : str = 'binary') -> float:
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    def _tp_fp_fn(label):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        return tp, fp, fn

    recalls = []
    for lbl in labels:
        tp, _, fn = _tp_fp_fn(lbl)
        if (tp + fn):
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        recalls.append(recall)

    if average == 'binary':
        return recalls[-1]
    elif average == 'macro':
        return np.mean(recalls)
    else:
        raise ValueError("average must be 'binary' or 'macro'")
    
def f1_score(y_true: Arr, y_pred: Arr, average : str = 'binary') -> float:
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    def _tp_fp_fn(label):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        return tp, fp, fn

    f1s = []
    for lbl in labels:
        tp, fp ,fn = _tp_fp_fn(lbl)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = precision = tp / (tp + fp) if (tp + fp) else 0.0

        if (precision + recall):
            f1 = (2*precision*recall)/(precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)

    if average == 'binary':
        return f1s[-1]
    elif average == 'macro':
        return np.mean(f1s)
    else:
        raise ValueError("average must be 'binary' or 'macro'")

def confusion_matrix(y_true: Arr, y_pred: Arr) -> np.ndarray:
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels
