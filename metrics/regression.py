import numpy as np
from core.tensor import Tensor
from typing import Union

Arr = Union[Tensor, np.ndarray]

def _to_numpy(x):
    if isinstance(x, Tensor):
        return x.data
    return np.array(x)

# MSE
def mean_squared_error(y_true: Arr, y_pred: Arr) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# RMSE
def root_mean_squared_error(y_true: Arr, y_pred: Arr) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

# MAE
def mean_absolute_error(y_true: Arr, y_pred: Arr) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return np.mean(np.abs(y_true - y_pred))

# R-squared error
def r2_score(y_true: Arr, y_pred: Arr) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot