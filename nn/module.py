from core.tensor import Tensor

class BaseModule:
    def __init__(self):
        self._params = []
        self.training = True

    def parameters(self):
        return self._params

    def add_param(self, param):
        self._params.append(param)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False