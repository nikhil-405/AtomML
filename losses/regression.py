# pytest tests/test_losses.py
import numpy as np
from typing import Union, Sequence
from core.tensor import Tensor

Arr = Union[Tensor, np.ndarray, Sequence[float]]

def _to_tensor(x: Arr) -> Tensor:
    if isinstance(x, Tensor):
        return x
    arr = np.array(x, dtype=float)
    return Tensor(arr, requires_grad=False)

# MSE
def MSE(y_true: Arr, y_pred: Arr) -> Tensor:
    yt = _to_tensor(y_true)
    yp = _to_tensor(y_pred)
    diff = yt - yp
    return (diff * diff).sum() * (1.0 / yt.data.shape[0])

# RMSE 
def RMSE(y_true: Arr, y_pred: Arr) -> Tensor:
    mse = MSE(y_true, y_pred)
    return mse ** Tensor(0.5, requires_grad = False)

# MAE
def MAE(y_true: Arr, y_pred: Arr) -> Tensor:
    yt = _to_tensor(y_true)
    yp = _to_tensor(y_pred)
    diff = yt - yp
    # abs(x) = ReLU(x) + ReLU(-x)
    abs_diff = diff.relu() + (-diff).relu()
    return abs_diff.sum() * (1.0 / yt.data.shape[0])

# R-squared loss
def R2(y_true: Arr, y_pred: Arr) -> Tensor:
    yt = _to_tensor(y_true)
    yp = _to_tensor(y_pred)
    res = (yt - yp) ** Tensor(2.0, requires_grad=False)
    ss_res = res.sum()
    mean_true = Tensor(float(np.mean(yt.data)), requires_grad = False)
    tot = ((yt - mean_true) ** Tensor(2.0, requires_grad = False)).sum()
    return Tensor(1.0, requires_grad=False) - ss_res / tot