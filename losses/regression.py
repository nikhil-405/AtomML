import numpy as np
from core.tensor import Tensor
from typing import Union

Arr = Union[Tensor, np.ndarray]

def _to_numpy(x):
    if isinstance(x, Tensor):
        return x.data
    return np.array(x)

# MSE
def MSE(y_true: Tensor, y_pred: Tensor) -> Tensor:
    diff = y_true - y_pred
    loss = (diff * diff).sum() * (1.0 / y_true.data.shape[0])
    return loss

# RMSE
def RMSE(y_true: Arr, y_pred: Arr) -> Tensor:
    return Tensor(np.sqrt(MSE(y_true, y_pred)), requires_grad = True)

# MAE
def MAE(y_true: Arr, y_pred: Arr) -> Tensor:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return Tensor(np.mean(np.abs(y_true - y_pred)), requires_grad = True)

# R-squared error
def R2(y_true: Arr, y_pred: Arr) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return Tensor(1 - ss_res / ss_tot, requires_grad = True)