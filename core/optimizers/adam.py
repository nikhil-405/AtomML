import numpy as np
from ..tensor import Tensor
from .base import Optimizer

class Adam(Optimizer):
    """ Adaptive Moment Optimizer"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {p: np.zeros_like(p.data) for p in self.params}
        self.v = {p: np.zeros_like(p.data) for p in self.params}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad

            m_new = self.beta1 * self.m[p] + (1 - self.beta1) * g
            v_new = self.beta2 * self.v[p] + (1 - self.beta2) * (g * g)

            m_hat = m_new / (1 - self.beta1**self.t)
            v_hat = v_new / (1 - self.beta2**self.t)

            p.data -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

            self.m[p] = m_new
            self.v[p] = v_new