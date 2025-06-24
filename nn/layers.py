import numpy as np
from core.tensor import Tensor
from nn.module import BaseModule

class Linear(BaseModule):
    def __init__(self, _in: int, _out: int):
        super().__init__()

        w = Tensor(np.random.randn(_in, _out), requires_grad = True)
        b = Tensor(np.zeros(_out), requires_grad = True)
    
        self._in = _in
        self._out = _out
        self.w = w
        self.b = b
        self.add_param(w)
        self.add_param(b)

    def forward(self, x):
        return x @ self.w + self.b