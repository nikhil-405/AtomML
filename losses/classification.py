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

# Binary Cross-Entropy Loss
def bce_loss(y_true: Union[Tensor,np.ndarray], y_pred: Union[Tensor,np.ndarray]) -> Tensor:
    yt = _to_tensor(y_true)
    yp = _to_tensor(y_pred)
    eps = Tensor(1e-8, requires_grad = False)
    one = Tensor(1.0, requires_grad=False)

    term1 = yt * (yp + eps).log()
    term2 = (one - yt) * (one - yp + eps).log()

    return - (term1 + term2).sum() * (1.0 / yt.data.shape[0])

# Cross-Entropy Loss
def ce_loss(y_true: Arr, y_pred: Tensor) -> Tensor:
    logits = y_pred 
    probs = logits.softmax(dim = 1)

    yt = _to_tensor(y_true)
    labels = yt.data.flatten().astype(int)

    # print("logits.shape:", logits.data.shape)
    # print("probs.shape:", probs.data.shape)

    N, C = probs.data.shape

    one_hot = np.zeros((N, C), dtype=float)
    one_hot[np.arange(N), labels] = 1.0
    yt = Tensor(one_hot, requires_grad=False)

    eps = Tensor(1e-8, requires_grad=False)
    log_probs = (probs + eps).log()
    loss = - (yt * log_probs).sum() * (1.0 / N)
    return loss