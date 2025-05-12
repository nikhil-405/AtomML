import numpy as np
from ..tensor import Tensor
from .base import Optimizer

class SGD(Optimizer):
    """Stochastic Gradient Descent Optimizer"""
    def __init__(self, params, lr = 1e-3, momentum = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {p: 0 for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            if self.momentum:
                v_prev = self.velocities[p]
                v_new = self.momentum * v_prev + (1 - self.momentum) * p.grad
                self.velocities[p] = v_new
                update = v_new
            else:
                update = p.grad
            p.data -= self.lr * update
